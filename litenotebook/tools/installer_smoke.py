"""Clean-install test of a release installer, on the platform that runs it.

    python tools/installer_smoke.py linux   DIST
    python tools/installer_smoke.py windows DIST [--old-installer OLD.exe]

The cycle: install → launch → create data → quit → install again (and, on
Windows with --old-installer, upgrade from that older version) → launch →
data still there → uninstall → program, launcher, shortcuts and uninstall
entry gone, data kept.

Linux runs install-litenotebook.sh with a temporary HOME whose path has a
space and non-ASCII characters (".../home/test user José"); nothing outside
it is touched. Windows runs the real Setup wizard silently, per user, into a
program folder with a space and "José" in its name, and the app with a data
folder of the same kind; it changes this Windows account (installs, then
uninstalls), so run it on a CI runner or a test machine.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

RESULTS: list[tuple[bool, str]] = []


def check(ok: object, label: str) -> bool:
    RESULTS.append((bool(ok), label))
    print(("PASS " if ok else "FAIL ") + label, flush=True)
    return bool(ok)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def request(
    base: str, method: str, path: str, data: dict[str, str] | None = None
) -> bytes:
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    req = urllib.request.Request(
        base + path,
        data=body,
        method=method,
        headers={"Origin": base, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return bytes(r.read())


class App:
    """The installed app on a free port; Quit through its own Quit action."""

    def __init__(self, cmd: list[str], env: dict[str, str]) -> None:
        port = free_port()
        self.base = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen(
            [*cmd, "--no-browser", "--port", str(port)],
            env={**env, "BROWSER": "true"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(600):
            try:
                urllib.request.urlopen(self.base + "/livez", timeout=1)
                return
            except OSError:
                if self.proc.poll() is not None:
                    break
                time.sleep(0.1)
        raise SystemExit("the installed LiteNotebook did not start")

    def quit(self) -> int:
        request(self.base, "POST", "/app/quit", {})
        return self.proc.wait(60)


# --- Linux ---------------------------------------------------------------------------


def linux(dist: Path) -> None:
    work = Path(tempfile.mkdtemp(prefix="ln-install-"))
    home = work / "home" / "test user José"
    home.mkdir(parents=True)
    env = {
        "HOME": str(home),
        "PATH": f"{home}/.local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
    }
    script = dist / "install-litenotebook.sh"

    def install(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), *args],
            env=env,
            cwd=dist,
            capture_output=True,
            text=True,
            timeout=900,
        )

    data = home / ".local/share/LiteNotebook"
    prefix = home / ".local/opt/litenotebook"
    launcher = home / ".local/bin/litenotebook"
    try:
        r = install("--yes")
        check(
            r.returncode == 0 and "installed successfully" in r.stdout,
            "install (--yes)",
        )
        check(
            sorted(p.name for p in home.iterdir()) == [".local"],
            "only ~/.local was written",
        )
        app = App([str(launcher)], env)
        request(app.base, "POST", "/notebooks", {"name": "Kept across upgrades"})
        check(
            (data / "litenotebook.sqlite3").is_file(),
            "data created in ~/.local/share/LiteNotebook",
        )
        check(app.quit() == 0, "Quit")
        runtime = any(dist.glob("LiteNotebook-embedding-runtime-*.tar.gz"))
        args = ["--yes", "--with-embedding-runtime"] if runtime else ["--yes"]
        r = install(*args)
        check(
            r.returncode == 0,
            "install again over it" + (" with the runtime" if runtime else ""),
        )
        if runtime:
            check(
                (prefix / "embedding-runtime/litenotebook-embed-worker").is_file(),
                "runtime installed beside the app",
            )
        app = App([str(launcher)], env)
        check(
            b"Kept across upgrades" in request(app.base, "GET", "/"), "data still there"
        )
        check(app.quit() == 0, "Quit again")
        r = subprocess.run(
            [str(home / ".local/bin/litenotebook-uninstall"), "--yes"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        check(
            r.returncode == 0 and str(data) in r.stdout,
            "uninstall names the kept data folder",
        )
        gone = [
            prefix,
            launcher,
            home / ".local/share/applications/litenotebook.desktop",
        ]
        check(
            not any(p.exists() for p in gone),
            "program, launcher and menu entry removed",
        )
        check(
            not list((home / ".local/share/icons").rglob("litenotebook.*")),
            "icons removed",
        )
        check((data / "litenotebook.sqlite3").is_file(), "notebook data kept")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- Windows -------------------------------------------------------------------------

UNINSTALL_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    r"\{6C1E9E5A-2C3B-4F35-9A7E-0B7D2C9F4A11}_is1"
)


def _display_version() -> str | None:
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as key:
            return str(winreg.QueryValueEx(key, "DisplayVersion")[0])
    except OSError:
        return None


def windows(dist: Path, old_installer: Path | None) -> None:
    manifest = json.loads((dist / "release-manifest.json").read_text("utf-8"))
    version = manifest["version"]
    setup = dist / f"LiteNotebook-Setup-{version}-x64.exe"
    work = Path(tempfile.mkdtemp(prefix="ln-install-"))
    target = work / "Test User José" / "LiteNotebook"
    data = work / "Test User José" / "data"
    start_menu = (
        Path(os.environ["APPDATA"])
        / "Microsoft/Windows/Start Menu/Programs/LiteNotebook"
    )
    desktop = Path(os.environ["USERPROFILE"]) / "Desktop" / "LiteNotebook.lnk"
    env = dict(os.environ)

    def run_setup(exe: Path, *extra: str) -> int:
        cmd = [
            str(exe),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            f"/DIR={target}",
            *extra,
        ]
        return subprocess.run(cmd, timeout=1800).returncode

    app_cmd = [str(target / "LiteNotebook.exe"), "--data-dir", str(data)]
    try:
        if old_installer is not None:
            check(
                run_setup(old_installer, "/COMPONENTS=app") == 0,
                f"older installer {old_installer.name}",
            )
            check(
                _display_version() not in (None, version),
                f"older version registered ({_display_version()})",
            )
        r = run_setup(
            setup, "/COMPONENTS=app,runtime", "/TASKS=startmenuicon,desktopicon"
        )
        check(
            r == 0,
            f"{setup.name}: silent per-user install"
            + (" (upgrade)" if old_installer else ""),
        )
        check(_display_version() == version, f"uninstall entry shows {version}")
        check(
            (target / "LiteNotebook.exe").is_file(),
            "program installed in the chosen folder (space + José)",
        )
        check(
            (target / "embedding-runtime/litenotebook-embed-worker.exe").is_file(),
            "optional runtime installed beside the app",
        )
        check((start_menu / "LiteNotebook.lnk").is_file(), "Start Menu shortcut")
        check(desktop.is_file(), "Desktop shortcut (when chosen)")
        app = App(app_cmd, env)
        request(app.base, "POST", "/notebooks", {"name": "Kept across upgrades"})
        check(app.quit() == 0, "launch, create data, Quit")
        check(
            run_setup(setup) == 0,
            "install the same version again (repair/upgrade path)",
        )
        check(
            (target / "embedding-runtime").is_dir(),
            "the earlier component choice is kept",
        )
        app = App(app_cmd, env)
        check(
            b"Kept across upgrades" in request(app.base, "GET", "/"), "data still there"
        )
        check(app.quit() == 0, "Quit again")
        uninstaller = target / "unins000.exe"
        r = subprocess.run(
            [str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
            timeout=600,
        ).returncode
        for _ in range(60):  # the uninstaller finishes in a copy of itself
            if not (target / "LiteNotebook.exe").exists():
                break
            time.sleep(1)
        check(
            r == 0 and not (target / "LiteNotebook.exe").exists(),
            "uninstall removes the program",
        )
        check(
            not (start_menu / "LiteNotebook.lnk").exists() and not desktop.exists(),
            "shortcuts removed",
        )
        check(_display_version() is None, "uninstall entry removed")
        check((data / "litenotebook.sqlite3").is_file(), "notebook data kept")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("platform", choices=["linux", "windows"])
    ap.add_argument("dist", type=Path)
    ap.add_argument("--old-installer", type=Path, help="Windows: upgrade from this")
    args = ap.parse_args()
    if args.platform == "windows":
        if sys.platform != "win32":
            raise SystemExit("run the windows check on Windows")
        windows(args.dist.resolve(), args.old_installer)
    else:
        linux(args.dist.resolve())
    failed = [label for ok, label in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
