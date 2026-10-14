"""Build LiteNotebook desktop packages for the current platform.

    uv sync --group packaging          # once: pinned PyInstaller
    uv run --group packaging python tools/build.py [--skip-worker] [--skip-appimage]

Output (ignored by git) under ``dist/litenotebook/``::

    LiteNotebook-<version>-<platform>/            the app folder (core, no torch)
    LiteNotebook-<version>-<platform>.tar.gz      portable archive (.zip on Windows)
    LiteNotebook-<version>-x86_64.AppImage        Linux AppImage
    LiteNotebook-embedding-runtime-<version>-<platform>.tar.gz   optional worker
    LiteNotebook-<version>-Setup.exe              Windows installer (on Windows,
                                                  when Inno Setup is installed)
    SHA256SUMS, build-report.json

Nothing optional is bundled: no GGUF models, no MiniLM files, no llama.cpp,
no speech engine or voices, no user data. Linux builds need a glibc at least
as new as the build machine's (build on an older distribution, e.g. the CI
workflow's Ubuntu 22.04 runner, for wide compatibility).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from litenotebook import __version__  # noqa: E402

PYINSTALLER_VERSION = "6.22.3"
APPIMAGETOOL = {
    "url": "https://github.com/AppImage/appimagetool/releases/download/1.9.1/"
    "appimagetool-x86_64.AppImage",
    "sha256": "ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0",
    "size": 15092216,
}
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
IS_WINDOWS = sys.platform == "win32"
BUILD = ROOT / "build" / "package"
DIST = ROOT / "dist" / "litenotebook"


def say(message: str) -> None:
    print(f"[build] {message}", flush=True)


def platform_tag() -> str:
    arch = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(platform.machine().lower(), platform.machine())
    return f"{'windows' if IS_WINDOWS else sys.platform}-{arch}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tree_size(path: Path) -> int:
    return sum(
        p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()
    )


def preflight() -> None:
    import PyInstaller

    if PyInstaller.__version__ != PYINSTALLER_VERSION:
        raise SystemExit(
            f"PyInstaller {PYINSTALLER_VERSION} is pinned; found "
            f"{PyInstaller.__version__}. "
            "Run: uv sync --group packaging"
        )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if f'version = "{__version__}"' not in pyproject:
        raise SystemExit("pyproject.toml version and litenotebook.__version__ differ")


# --- build metadata -------------------------------------------------------------


def git_commit() -> str:
    """Commit of the source tree (+dirty when LiteNotebook files changed).
    ``LN_GIT_COMMIT`` overrides it (container builds without .git)."""
    forced = os.environ.get("LN_GIT_COMMIT", "").strip()
    if forced:
        return forced
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head}+dirty" if dirty else head


def runner_info() -> dict[str, str]:
    """Where the build ran — OS and versions only, no host or user names."""
    info = {
        "os": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
    }
    if sys.platform.startswith("linux"):
        libc, version = platform.libc_ver()
        info["libc"] = f"{libc} {version}".strip()
        try:
            for line in Path("/etc/os-release").read_text("utf-8").splitlines():
                if line.startswith("PRETTY_NAME="):
                    info["distribution"] = line.split("=", 1)[1].strip('"')
        except OSError:
            pass
    if IS_WINDOWS:
        info["windows"] = platform.win32_ver()[0] + " " + platform.win32_ver()[1]
    for key in (
        "GITHUB_ACTIONS",
        "RUNNER_OS",
        "ImageOS",
        "ImageVersion",
        "LN_BUILD_RUNNER",
    ):
        if os.environ.get(key):
            info[key.lower()] = os.environ[key]
    return info


_GLIBC = re.compile(rb"GLIBC_2\.(\d+)(?:\.(\d+))?")
_GLIBCXX = re.compile(rb"GLIBCXX_3\.4\.(\d+)")


def glibc_requirement(folder: Path) -> dict[str, object]:
    """Newest GLIBC / GLIBCXX symbol versions referenced by any ELF file in
    the folder: the minimum C library a target system needs."""
    worst = (2, 0, 0)
    worst_files: list[str] = []
    cxx = 0
    for path in folder.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            with open(path, "rb") as f:
                if f.read(4) != b"\x7fELF":
                    continue
                data = f.read()
        except OSError:
            continue
        versions = [(2, int(a), int(b or 0)) for a, b in _GLIBC.findall(data)]
        if versions:
            top = max(versions)
            rel = path.relative_to(folder).as_posix()
            if top > worst:
                worst, worst_files = top, [rel]
            elif top == worst:
                worst_files.append(rel)
        cxx = max([cxx, *(int(v) for v in _GLIBCXX.findall(data))])
    text = f"{worst[0]}.{worst[1]}" + (f".{worst[2]}" if worst[2] else "")
    return {
        "min_glibc": text,
        "needed_by": sorted(worst_files)[:8],
        "max_glibcxx": f"3.4.{cxx}" if cxx else None,
    }


# --- icons ------------------------------------------------------------------------


def make_icons(out: Path) -> dict[str, Path]:
    """The app icon as PNG sizes and a Windows .ico, pre-rendered from the SVG
    into packaging/icons/ (a PDF engine cannot render SVG, and the build
    must not need another graphics library). The recorded SVG checksum
    makes a stale icon set fail the build instead of shipping silently."""
    icons = ROOT / "packaging" / "icons"
    svg = ROOT / "src" / "litenotebook" / "web" / "static" / "icon.svg"
    recorded = (icons / "icon.svg.sha256").read_text("utf-8").split()[0]
    if hashlib.sha256(svg.read_bytes()).hexdigest() != recorded:
        raise SystemExit(
            "icon.svg changed: re-render packaging/icons/litenotebook-<size>.png "
            f"for sizes {ICON_SIZES} and litenotebook.ico, then update "
            "packaging/icons/icon.svg.sha256"
        )
    out.mkdir(parents=True, exist_ok=True)
    for size in ICON_SIZES:
        shutil.copy(icons / f"litenotebook-{size}.png", out)
    shutil.copy(icons / "litenotebook.ico", out)
    shutil.copy(svg, out / "litenotebook.svg")
    return {
        "ico": out / "litenotebook.ico",
        "png": out / "litenotebook-256.png",
        "svg": out / "litenotebook.svg",
    }


def copy_notice_files(target: Path) -> None:
    """Licence texts that must travel with bundled native code: PDFium and
    the libraries compiled into it (shipped inside the pypdfium2 wheel)."""
    from importlib import metadata

    dist = metadata.distribution("pypdfium2")
    files = [f for f in dist.files or [] if "/licenses/" in f"/{f}"]
    if not files:
        raise SystemExit("pypdfium2 licence files not found")
    for item in files:
        source = Path(str(dist.locate_file(item)))
        rel = str(item).split("/licenses/", 1)[1]
        dest = target / "pypdfium2" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)


# --- PyInstaller ------------------------------------------------------------------


def pyinstaller(spec: str, env: dict[str, str]) -> Path:
    work = BUILD / "work"
    dist = BUILD / "pyinstaller"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        str(ROOT / "packaging" / "pyinstaller" / spec),
    ]
    log = BUILD / f"{spec}.log"
    say(f"PyInstaller {spec} (log: {log})")
    started = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            env={**os.environ, **env},
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(f"PyInstaller failed for {spec}; see {log}")
    say(f"  done in {time.perf_counter() - started:.0f} s")
    return dist


# --- archives ---------------------------------------------------------------------


def archive(folder: Path, target: Path, arcname: str) -> Path:
    say(f"Archiving {target.name}")
    if target.suffix == ".zip":
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(folder.rglob("*")):
                zf.write(p, f"{arcname}/{p.relative_to(folder).as_posix()}")
    else:
        with tarfile.open(target, "w:gz", compresslevel=6) as tf:
            tf.add(folder, arcname=arcname)
    return target


def appimage(app_folder: Path, icons: dict[str, Path], out: Path) -> Path:
    tools = BUILD / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    tool = tools / "appimagetool-1.9.1-x86_64.AppImage"
    if not tool.exists() or sha256(tool) != APPIMAGETOOL["sha256"]:
        say("Downloading the pinned appimagetool 1.9.1")
        partial = tool.with_suffix(".part")
        urllib.request.urlretrieve(APPIMAGETOOL["url"], partial)  # noqa: S310 - pinned https
        if (
            partial.stat().st_size != APPIMAGETOOL["size"]
            or sha256(partial) != APPIMAGETOOL["sha256"]
        ):
            partial.unlink()
            raise SystemExit("appimagetool download does not match its pinned checksum")
        partial.replace(tool)
        tool.chmod(0o755)
    appdir = BUILD / "LiteNotebook.AppDir"
    if appdir.exists():
        shutil.rmtree(appdir)
    target = appdir / "usr" / "lib" / "LiteNotebook"
    shutil.copytree(app_folder, target, symlinks=True)
    (appdir / "AppRun").symlink_to("usr/lib/LiteNotebook/litenotebook")
    desktop = (ROOT / "packaging" / "linux" / "litenotebook.desktop").read_text(
        encoding="utf-8"
    )
    (appdir / "litenotebook.desktop").write_text(desktop, encoding="utf-8")
    shutil.copy(icons["png"], appdir / "litenotebook.png")
    (appdir / ".DirIcon").symlink_to("litenotebook.png")
    say(f"Building {out.name}")
    env = {**os.environ, "ARCH": "x86_64", "APPIMAGE_EXTRACT_AND_RUN": "1"}
    result = subprocess.run(
        [str(tool), "--no-appstream", str(appdir), str(out)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"appimagetool failed: {result.stderr[-2000:]}")
    return out


def inno_setup(
    app_folder: Path, runtime_folder: Path | None, icons: dict[str, Path]
) -> Path | None:
    candidates = [
        shutil.which("iscc"),
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    iscc = next((c for c in candidates if c and Path(c).exists()), None)
    if iscc is None:
        say("Inno Setup (ISCC) not found: skipping the Windows installer")
        return None
    say("Building the Windows installer (Inno Setup)")
    defines = [
        f"/DAppVersion={__version__}",
        f"/DAppFolder={app_folder}",
        f"/DOutputDir={DIST}",
        f"/DIconFile={icons['ico']}",
    ]
    if runtime_folder is not None:
        defines.append(f"/DRuntimeFolder={runtime_folder}")
    script = ROOT / "packaging" / "windows" / "litenotebook.iss"
    subprocess.run([iscc, *defines, str(script)], check=True)
    out = DIST / f"LiteNotebook-{__version__}-Setup.exe"
    return out if out.exists() else None


# --- main -------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-worker",
        action="store_true",
        help="do not build the optional embedding runtime",
    )
    parser.add_argument("--skip-appimage", action="store_true")
    args = parser.parse_args()
    preflight()
    started = time.perf_counter()
    tag = platform_tag()
    BUILD.mkdir(parents=True, exist_ok=True)
    DIST.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    report: dict[str, object] = {
        "app": "LiteNotebook",
        "version": __version__,
        "platform": tag,
        "architecture": tag.rsplit("-", 1)[1],
        "git_commit": commit,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pyinstaller": PYINSTALLER_VERSION,
        "runner": runner_info(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    icons = make_icons(BUILD / "icons")
    from licenses import write_inventory  # tools/licenses.py

    inventory = write_inventory(BUILD / "THIRD_PARTY_LICENSES.md")
    env = {
        "LN_ROOT": str(ROOT),
        "LN_ICON": str(icons["ico"]) if IS_WINDOWS else "",
        "LN_APP_NAME": "LiteNotebook" if IS_WINDOWS else "litenotebook",
    }

    runtime_entry = None
    runtime_folder: Path | None = None
    if not args.skip_worker:
        built = pyinstaller("embed_worker.spec", env) / "litenotebook-embed-worker"
        runtime_folder = BUILD / "embedding-runtime"
        if runtime_folder.exists():
            shutil.rmtree(runtime_folder)
        shutil.copytree(built, runtime_folder, symlinks=True)
        ext = ".zip" if IS_WINDOWS else ".tar.gz"
        name = f"LiteNotebook-embedding-runtime-{__version__}-{tag}{ext}"
        runtime_archive = archive(runtime_folder, DIST / name, "embedding-runtime")
        runtime_entry = {
            "file": name,
            "size": runtime_archive.stat().st_size,
            "sha256": sha256(runtime_archive),
            "unpacked_size": tree_size(runtime_folder),
        }
        report["embedding_runtime"] = runtime_entry

    core_dist = pyinstaller("litenotebook.spec", env) / env["LN_APP_NAME"]
    app_name = f"LiteNotebook-{__version__}-{tag}"
    app_folder = DIST / app_name
    if app_folder.exists():
        shutil.rmtree(app_folder)
    shutil.copytree(core_dist, app_folder, symlinks=True)
    shutil.copy(icons["png"], app_folder / "litenotebook.png")
    shutil.copy(icons["svg"], app_folder / "litenotebook.svg")
    if not IS_WINDOWS:
        shutil.copy(
            ROOT / "packaging" / "linux" / "litenotebook.desktop",
            app_folder / "litenotebook.desktop",
        )
    shutil.copy(
        BUILD / "THIRD_PARTY_LICENSES.md", app_folder / "THIRD_PARTY_LICENSES.md"
    )
    shutil.copy(ROOT / "packaging" / "README-PACKAGE.txt", app_folder / "README.txt")
    copy_notice_files(app_folder / "licenses")
    if (ROOT.parent / "LICENSE").is_file():
        shutil.copy(ROOT.parent / "LICENSE", app_folder / "LICENSE.txt")
    manifest = {
        "app": "LiteNotebook",
        "version": __version__,
        "platform": tag,
        "architecture": tag.rsplit("-", 1)[1],
        "git_commit": commit,
        "built_at": report["built_at"],
        "python": platform.python_version(),
        "pyinstaller": PYINSTALLER_VERSION,
        "embedding_runtime": runtime_entry,
        "bundled_optional_components": [],  # models, llama.cpp, speech: never
    }
    (app_folder / "litenotebook-build.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    report["app_folder_size"] = tree_size(app_folder)
    ext = ".zip" if IS_WINDOWS else ".tar.gz"
    core_archive = archive(app_folder, DIST / f"{app_name}{ext}", app_name)
    report["app_archive"] = {
        "file": core_archive.name,
        "size": core_archive.stat().st_size,
    }
    artifacts = [core_archive]
    if runtime_entry:
        artifacts.append(DIST / runtime_entry["file"])
    if not IS_WINDOWS and not args.skip_appimage:
        image = appimage(
            app_folder, icons, DIST / f"LiteNotebook-{__version__}-x86_64.AppImage"
        )
        report["appimage"] = {"file": image.name, "size": image.stat().st_size}
        artifacts.append(image)
    if IS_WINDOWS:
        setup = inno_setup(app_folder, runtime_folder, icons)
        if setup is not None:
            report["installer"] = {"file": setup.name, "size": setup.stat().st_size}
            artifacts.append(setup)
    if not IS_WINDOWS:
        report["glibc"] = {"app": glibc_requirement(app_folder)}
        if runtime_folder is not None:
            report["glibc"]["embedding_runtime"] = glibc_requirement(runtime_folder)  # type: ignore[index]
    digests = {p.name: sha256(p) for p in artifacts}
    report["artifacts"] = [
        {"file": p.name, "size": p.stat().st_size, "sha256": digests[p.name]}
        for p in artifacts
    ]
    sums = "".join(f"{digests[p.name]}  {p.name}\n" for p in artifacts)
    (DIST / "SHA256SUMS").write_text(sums, encoding="utf-8")
    report["licenses"] = {"packages": len(inventory)}
    report["build_seconds"] = round(time.perf_counter() - started)
    (DIST / "build-report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    say(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
