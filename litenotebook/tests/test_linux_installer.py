"""The per-user Linux installer (install-litenotebook.sh) against synthetic
release archives: install, upgrade, failed upgrade, uninstall with data kept,
optional runtime, and hostile or tampered archives. No root, no network."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from linux_installer import render  # noqa: E402

pytestmark = [
    pytest.mark.posix_only,
    pytest.mark.skipif(
        shutil.which("bash") is None or shutil.which("tar") is None,
        reason="needs bash and tar",
    ),
]

SLEEP = shutil.which("sleep") or "/bin/sleep"  # a real ELF for "running" checks


def _add(tar: tarfile.TarFile, name: str, data: bytes = b"", mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, int(time.time())
    tar.addfile(info, io.BytesIO(data))


def _dir(tar: tarfile.TarFile, name: str) -> None:
    info = tarfile.TarInfo(name)
    info.type, info.mode = tarfile.DIRTYPE, 0o755
    tar.addfile(info)


def make_core(
    path: Path,
    version: str,
    *,
    extra: list[tarfile.TarInfo] | None = None,
    top: str | None = None,
    program: bool = True,
    build_version: str | None = None,
) -> dict[str, Any]:
    top = top or f"LiteNotebook-{version}-linux-x86_64"
    with tarfile.open(path, "w:gz") as tar:
        _dir(tar, top)
        if program:
            _add(tar, f"{top}/litenotebook", Path(SLEEP).read_bytes(), 0o755)
        _dir(tar, f"{top}/_internal")
        _add(tar, f"{top}/_internal/libdemo.so.1", b"lib")
        link = tarfile.TarInfo(f"{top}/_internal/libdemo.so")
        link.type, link.linkname = tarfile.SYMTYPE, "libdemo.so.1"
        tar.addfile(link)
        _add(tar, f"{top}/_internal/Lorem ipsum.txt", b"text")
        manifest = {"app": "LiteNotebook", "version": build_version or version}
        _add(
            tar,
            f"{top}/litenotebook-build.json",
            json.dumps(manifest, indent=1).encode(),
        )
        _add(tar, f"{top}/litenotebook.png", b"\x89PNG demo")
        _add(tar, f"{top}/litenotebook.svg", b"<svg/>")
        for info in extra or []:
            tar.addfile(info, io.BytesIO(b"x" * info.size) if info.isfile() else None)
    return {
        "file": path.name,
        "size": path.stat().st_size,
        "sha256": _sha(path),
        "dir": top,
    }


def make_runtime(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "w:gz") as tar:
        _dir(tar, "embedding-runtime")
        _add(tar, "embedding-runtime/litenotebook-embed-worker", b"#!/bin/sh\n", 0o755)
        _dir(tar, "embedding-runtime/_internal")
    return {"file": path.name, "size": path.stat().st_size, "sha256": _sha(path)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Release:
    """A folder with the archives and their installer, and a fake home with a
    space and non-ASCII characters in its path."""

    def __init__(self, tmp: Path, name: str = "rel") -> None:
        self.dir = tmp / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.home = tmp / "home" / "test user José"
        self.home.mkdir(parents=True, exist_ok=True)

    def build(
        self, version: str, runtime: bool = False, **core_kw: Any
    ) -> dict[str, Any]:
        for old in self.dir.iterdir():
            old.unlink()
        core = make_core(
            self.dir / f"LiteNotebook-{version}-linux-x86_64.tar.gz", version, **core_kw
        )
        rt = None
        if runtime:
            rt = make_runtime(
                self.dir
                / f"LiteNotebook-embedding-runtime-{version}-linux-x86_64.tar.gz"
            )
        script = self.dir / "install-litenotebook.sh"
        script.write_text(render(version, core, rt), encoding="utf-8")
        script.chmod(0o755)
        return core

    def env(self) -> dict[str, str]:
        return {
            "HOME": str(self.home),
            "PATH": f"{self.home}/.local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
        }

    def run(
        self, *args: str, stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.dir / "install-litenotebook.sh"), *args],
            env=self.env(),
            cwd=self.dir,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=120,
        )

    @property
    def prefix(self) -> Path:
        return self.home / ".local" / "opt" / "litenotebook"

    @property
    def data(self) -> Path:
        return self.home / ".local" / "share" / "LiteNotebook"

    def installed_version(self) -> str:
        info = (self.prefix / ".litenotebook-install").read_text("utf-8")
        return dict(line.split("=", 1) for line in info.splitlines())["version"]


@pytest.fixture
def rel(tmp_path: Path) -> Release:
    return Release(tmp_path)


def test_install_puts_files_in_xdg_places_only(rel: Release) -> None:
    rel.build("9.9.1")
    r = rel.run("--yes")
    assert r.returncode == 0, r.stderr
    for line in (
        "Verifying LiteNotebook-9.9.1-linux-x86_64.tar.gz...",
        "Installing LiteNotebook...",
        "Installing desktop launcher...",
        "Installing icon...",
        "LiteNotebook 9.9.1 installed successfully.",
    ):
        assert line in r.stdout
    assert sorted(p.name for p in rel.home.iterdir()) == [".local"]  # nothing in ~
    launcher = rel.home / ".local/bin/litenotebook"
    assert subprocess.run([str(launcher), "0"], timeout=10).returncode == 0
    desktop = (rel.home / ".local/share/applications/litenotebook.desktop").read_text(
        "utf-8"
    )
    assert f'Exec="{rel.prefix}/litenotebook"' in desktop  # quoted: space and é
    assert (
        rel.home / ".local/share/icons/hicolor/256x256/apps/litenotebook.png"
    ).is_file()
    assert (
        rel.home / ".local/share/icons/hicolor/scalable/apps/litenotebook.svg"
    ).is_file()
    assert (rel.prefix / "_internal/libdemo.so").is_symlink()
    assert not (rel.prefix / "embedding-runtime").exists()
    assert not list(rel.prefix.parent.glob(".litenotebook-*"))  # staging cleaned
    if shutil.which("desktop-file-validate"):
        v = subprocess.run(
            [
                "desktop-file-validate",
                str(rel.home / ".local/share/applications/litenotebook.desktop"),
            ],
            capture_output=True,
            text=True,
        )
        assert v.returncode == 0, v.stdout


def test_options_desktop_shortcut_and_no_menu_entry(rel: Release) -> None:
    (rel.home / "Desktop").mkdir()
    rel.build("9.9.1")
    r = rel.run("--yes", "--no-desktop-entry", "--desktop-shortcut")
    assert r.returncode == 0, r.stderr
    assert not (rel.home / ".local/share/applications/litenotebook.desktop").exists()
    shortcut = rel.home / "Desktop/litenotebook.desktop"
    assert shortcut.is_file() and os.access(shortcut, os.X_OK)


def test_upgrade_keeps_data_and_replaces_the_program(rel: Release) -> None:
    rel.build("9.9.1", runtime=True)
    assert rel.run("--yes", "--with-embedding-runtime").returncode == 0
    assert (rel.prefix / "embedding-runtime/litenotebook-embed-worker").is_file()
    (rel.data / "notebooks" / "1").mkdir(parents=True)
    note = rel.data / "notebooks" / "1" / "notes.txt"
    note.write_text("my notes", encoding="utf-8")
    (rel.prefix / "stale-file-from-old-version").write_text("x", encoding="utf-8")

    rel.build("9.9.2", runtime=True)
    r = rel.run("--yes")  # runtime not asked for: the 9.9.1 runtime is not kept
    assert r.returncode == 0, r.stderr
    assert rel.installed_version() == "9.9.2"
    assert not (rel.prefix / "stale-file-from-old-version").exists()
    assert not (rel.prefix / "embedding-runtime").exists()
    assert "Semantic search runtime: not installed" in r.stdout
    assert note.read_text("utf-8") == "my notes"

    assert rel.run("--yes", "--with-embedding-runtime").returncode == 0
    r = rel.run("--yes")  # same version again: its runtime is kept
    assert "Keeping the installed semantic-search runtime" in r.stdout
    assert (rel.prefix / "embedding-runtime/litenotebook-embed-worker").is_file()


def test_failed_upgrade_leaves_the_old_install_working(rel: Release) -> None:
    rel.build("9.9.1")
    assert rel.run("--yes").returncode == 0
    rel.build("9.9.2", program=False)  # valid checksum, but no program inside
    r = rel.run("--yes")
    assert r.returncode != 0 and "no litenotebook program" in r.stderr
    assert rel.installed_version() == "9.9.1"
    launcher = rel.home / ".local/bin/litenotebook"
    assert subprocess.run([str(launcher), "0"], timeout=10).returncode == 0
    assert not list(rel.prefix.parent.glob(".litenotebook-*"))


def test_uninstall_removes_the_program_and_keeps_data(rel: Release) -> None:
    (rel.home / "Desktop").mkdir()
    rel.build("9.9.1")
    assert rel.run("--yes", "--desktop-shortcut").returncode == 0
    rel.data.mkdir(parents=True)
    (rel.data / "litenotebook.sqlite3").write_bytes(b"db")
    (rel.data / "backups").mkdir()
    r = subprocess.run(
        [str(rel.home / ".local/bin/litenotebook-uninstall"), "--yes"],
        env=rel.env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert str(rel.data) in r.stdout and "were kept" in r.stdout
    assert not rel.prefix.exists()
    for gone in (
        ".local/bin/litenotebook",
        ".local/bin/litenotebook-uninstall",
        ".local/share/applications/litenotebook.desktop",
        ".local/share/icons/hicolor/256x256/apps/litenotebook.png",
        ".local/share/icons/hicolor/scalable/apps/litenotebook.svg",
        "Desktop/litenotebook.desktop",
    ):
        assert not (rel.home / gone).exists(), gone
    assert (rel.data / "litenotebook.sqlite3").read_bytes() == b"db"
    assert (rel.data / "backups").is_dir()


def test_interactive_questions_and_refusal_without_a_terminal(rel: Release) -> None:
    rel.build("9.9.1", runtime=True)
    r = rel.run()  # no terminal and no --yes
    assert r.returncode != 0 and "--yes" in r.stderr
    assert not rel.prefix.exists()


@pytest.mark.parametrize(
    "case",
    [
        "checksum",
        "size",
        "dotdot",
        "absolute",
        "top",
        "link_out",
        "link_abs",
        "hardlink",
        "device",
        "version",
        "name",
    ],
)
def test_hostile_or_tampered_archives_install_nothing(rel: Release, case: str) -> None:
    top = "LiteNotebook-9.9.1-linux-x86_64"
    extra: list[tarfile.TarInfo] = []

    def entry(
        name: str, kind: bytes = tarfile.REGTYPE, link: str = "", size: int = 1
    ) -> None:
        info = tarfile.TarInfo(name)
        info.type, info.linkname, info.size = (
            kind,
            link,
            size if kind == tarfile.REGTYPE else 0,
        )
        extra.append(info)

    kw: dict[str, Any] = {}
    if case == "dotdot":
        entry(f"{top}/../evil.sh")
    elif case == "absolute":
        entry("/tmp/evil.sh")
    elif case == "top":
        entry("other-folder/evil.sh")
    elif case == "link_out":
        entry(f"{top}/_internal/x", tarfile.SYMTYPE, "../../../../.bashrc")
    elif case == "link_abs":
        entry(f"{top}/_internal/x", tarfile.SYMTYPE, "/etc/passwd")
    elif case == "hardlink":
        entry(f"{top}/_internal/h", tarfile.LNKTYPE, f"{top}/litenotebook-build.json")
    elif case == "device":
        entry(f"{top}/_internal/dev", tarfile.CHRTYPE)
    elif case == "version":
        kw["build_version"] = "6.6.6"
    elif case == "name":
        entry(f'{top}/_internal/bad"quote')
    core = rel.build("9.9.1", extra=extra, **kw)
    archive = rel.dir / core["file"]
    if case == "checksum":  # same size, different bytes
        data = bytearray(archive.read_bytes())
        data[-5] ^= 0xFF
        archive.write_bytes(bytes(data))
    elif case == "size":
        with archive.open("ab") as f:
            f.write(b"\0")
    r = rel.run("--yes")
    assert r.returncode != 0, (case, r.stdout)
    assert not rel.prefix.exists()
    assert not (rel.home / ".local/bin/litenotebook").exists()
    assert not list(rel.home.parent.rglob("evil.sh"))
    assert (
        not list(rel.prefix.parent.glob(".litenotebook-*"))
        if rel.prefix.parent.exists()
        else True
    )


def test_refuses_foreign_folders_root_paths_and_the_data_folder(rel: Release) -> None:
    rel.build("9.9.1")
    foreign = rel.home / "Documents"
    foreign.mkdir()
    (foreign / "thesis.txt").write_text("keep", encoding="utf-8")
    r = rel.run("--yes", "--prefix", str(foreign))
    assert r.returncode != 0 and "not a LiteNotebook installation" in r.stderr
    assert (foreign / "thesis.txt").read_text("utf-8") == "keep"
    for bad in (str(rel.home), "relative/path", str(rel.data / "app")):
        r = rel.run("--yes", "--prefix", bad)
        assert r.returncode != 0, bad
    assert not (rel.data / "app").exists()


def test_upgrade_refused_while_litenotebook_runs(rel: Release) -> None:
    rel.build("9.9.1")
    assert rel.run("--yes").returncode == 0
    running = subprocess.Popen([str(rel.prefix / "litenotebook"), "30"])
    try:
        time.sleep(0.3)
        r = rel.run("--yes")
        assert r.returncode != 0 and "is running" in r.stderr
        u = rel.run("--uninstall", "--yes")
        assert u.returncode != 0 and rel.prefix.exists()
    finally:
        running.kill()
        running.wait()
    assert rel.run("--yes").returncode == 0


def test_template_refuses_to_run_unrendered(tmp_path: Path) -> None:
    template = ROOT / "packaging/linux/install-litenotebook.sh.in"
    r = subprocess.run(
        ["bash", str(template), "--yes"], capture_output=True, text=True, cwd=tmp_path
    )
    assert r.returncode != 0 and "template" in r.stderr


def test_render_rejects_unsafe_values() -> None:
    core = {"file": "a.tar.gz", "size": 1, "sha256": "ab"}
    with pytest.raises(ValueError):
        render("1.0$(id)", core, None)
    with pytest.raises(ValueError):
        render("1.0", {**core, "file": 'x".tar.gz'}, None)
