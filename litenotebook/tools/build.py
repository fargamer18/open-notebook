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
import shutil
import struct
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


# --- icons ------------------------------------------------------------------------


def make_icons(out: Path) -> dict[str, Path]:
    """PNG sizes and a Windows .ico from the original SVG (via PyMuPDF)."""
    import pymupdf

    out.mkdir(parents=True, exist_ok=True)
    svg = ROOT / "src" / "litenotebook" / "web" / "static" / "icon.svg"
    page = pymupdf.open(svg)[0]
    pngs: dict[int, bytes] = {}
    for size in ICON_SIZES:
        zoom = size / page.rect.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=True)
        data = pix.tobytes("png")
        (out / f"litenotebook-{size}.png").write_bytes(data)
        pngs[size] = data
    ico_sizes = [s for s in ICON_SIZES if s <= 256]
    header = struct.pack("<HHH", 0, 1, len(ico_sizes))
    offset = 6 + 16 * len(ico_sizes)
    entries, blobs = b"", b""
    for size in ico_sizes:
        data = pngs[size]
        dim = 0 if size == 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    (out / "litenotebook.ico").write_bytes(header + entries + blobs)
    shutil.copy(svg, out / "litenotebook.svg")
    return {
        "ico": out / "litenotebook.ico",
        "png": out / "litenotebook-256.png",
        "svg": out / "litenotebook.svg",
    }


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
    report: dict[str, object] = {
        "version": __version__,
        "platform": tag,
        "python": platform.python_version(),
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
    if (ROOT.parent / "LICENSE").is_file():
        shutil.copy(ROOT.parent / "LICENSE", app_folder / "LICENSE.txt")
    manifest = {
        "app": "LiteNotebook",
        "version": __version__,
        "platform": tag,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
    sums = "".join(f"{sha256(p)}  {p.name}\n" for p in artifacts)
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
