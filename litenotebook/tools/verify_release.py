"""Check a release build folder before anything is uploaded.

    python tools/verify_release.py dist/litenotebook

Checks, for the platform the build was made on (Linux or Windows):

- SHA256SUMS and release-manifest.json list the same files, and every file
  matches its size and SHA-256; artifact names are the deterministic ones;
- the app folder: the LiteNotebook executable, PDFium present, no PyMuPDF /
  MuPDF, graph.js and the vendored htmx present, no torch, no model files
  (GGUF, safetensors, ONNX, PyTorch), no database or notebook files, no
  path of the build machine's home or source tree in any text file;
- the embedding runtime archive holds the worker under embedding-runtime/;
- Linux: install-litenotebook.sh embeds the core archive's name and SHA-256.

Exit code 0 when everything passes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[bool, str]] = []
MODEL_SUFFIXES = (".gguf", ".safetensors", ".onnx", ".pt", ".pth", ".ckpt")
DATA_SUFFIXES = (".sqlite3", ".sqlite3-wal", ".faiss")


def check(ok: object, label: str) -> None:
    RESULTS.append((bool(ok), label))
    print(("PASS " if ok else "FAIL ") + label, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def archive_names(path: Path) -> list[str]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return z.namelist()
    with tarfile.open(path) as t:
        return t.getnames()


def main(dist: Path) -> int:
    manifest = json.loads((dist / "release-manifest.json").read_text("utf-8"))
    version = manifest["version"]
    artifacts = {a["filename"]: a for a in manifest["artifacts"]}
    sums = {}
    for line in (dist / "SHA256SUMS").read_text("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    check(
        set(sums) == set(artifacts),
        f"SHA256SUMS and the manifest list the same {len(sums)} files",
    )
    for name, entry in artifacts.items():
        path = dist / name
        ok = path.is_file() and path.stat().st_size == entry["size"]
        ok = ok and sha256(path) == entry["sha256"] == sums.get(name)
        check(ok, f"{name}: size and SHA-256 match")
        check(entry["version"] == version, f"{name}: version {entry['version']}")
    text = json.dumps(manifest)
    check(
        "/home/" not in text and "\\Users\\" not in text and str(ROOT) not in text,
        "manifest has no local paths",
    )

    platform = next(iter(artifacts.values()))["platform"]
    windows = platform == "windows"
    tag = "windows-x64" if windows else "linux-x86_64"
    ext = ".zip" if windows else ".tar.gz"
    expected = {
        f"LiteNotebook-{version}-{tag}{ext}": "app-archive",
        f"LiteNotebook-embedding-runtime-{version}-{tag}{ext}": "embedding-runtime",
    }
    if windows:
        expected[f"LiteNotebook-Setup-{version}-x64.exe"] = "windows-installer"
    else:
        expected[f"LiteNotebook-{version}-x86_64.AppImage"] = "appimage"
        expected["install-litenotebook.sh"] = "linux-installer"
    for name, kind in expected.items():
        check(
            artifacts.get(name, {}).get("component") == kind,
            f"artifact {name} ({kind})",
        )

    folder = dist / f"LiteNotebook-{version}-{tag}"
    files = [p for p in folder.rglob("*") if p.is_file() or p.is_symlink()]
    names = [p.relative_to(folder).as_posix().lower() for p in files]
    exe = "LiteNotebook.exe" if windows else "litenotebook"
    check((folder / exe).is_file(), f"app folder has {exe}")
    check(
        not any("mupdf" in n or "/fitz" in n or n.startswith("fitz") for n in names),
        "no PyMuPDF / MuPDF files",
    )
    pdfium = "pdfium.dll" if windows else "libpdfium.so"
    check(any(n.endswith(pdfium) for n in names), f"PDFium present ({pdfium})")
    check(
        any(
            n.endswith("licenses/pypdfium2") or "licenses/pypdfium2/" in n
            for n in names
        ),
        "PDFium licence notices present",
    )
    for asset in (
        "web/static/graph.js",
        "web/static/htmx.min.js",
        "web/static/htmx.license",
    ):
        check(
            any(n.endswith(asset) for n in names), f"bundled {asset.rsplit('/', 1)[1]}"
        )
    check(not any("/torch/" in f"/{n}" for n in names), "no torch in the core app")
    check(
        not any(n.endswith(MODEL_SUFFIXES) for n in names),
        "no model files in the core app",
    )
    check(
        not any(n.endswith(DATA_SUFFIXES) for n in names), "no database or index files"
    )
    leaks = []
    markers = [str(Path.home())] if len(str(Path.home())) > 5 else []
    if (ROOT / "pyproject.toml").is_file():  # running from the source checkout
        markers.append(str(ROOT))
    for p in files:
        if p.is_file() and p.suffix.lower() in (
            ".json",
            ".txt",
            ".md",
            ".desktop",
            ".cfg",
            ".toml",
            ".py",
        ):
            body = p.read_text("utf-8", errors="replace")
            leaks += [p.name for m in markers if m and m in body]
    check(not leaks, f"no build-machine paths in text files {leaks[:3]}")
    build = json.loads((folder / "litenotebook-build.json").read_text("utf-8"))
    check(
        build.get("version") == version
        and build.get("bundled_optional_components") == [],
        "build manifest: version, nothing optional bundled",
    )

    runtime = [n for n, a in artifacts.items() if a["component"] == "embedding-runtime"]
    if runtime:
        inside = archive_names(dist / runtime[0])
        worker = "embedding-runtime/litenotebook-embed-worker" + (
            ".exe" if windows else ""
        )
        check(worker in inside, "runtime archive holds the worker")
        check(
            all(n.split("/", 1)[0] == "embedding-runtime" for n in inside),
            "runtime archive has one top folder",
        )
    if not windows:
        script = (dist / "install-litenotebook.sh").read_text("utf-8")
        core = f"LiteNotebook-{version}-{tag}.tar.gz"
        check(
            f'CORE_FILE="{core}"' in script
            and f'CORE_SHA256="{artifacts[core]["sha256"]}"' in script,
            "installer script pins the core archive's SHA-256",
        )
    failed = [label for ok, label in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(Path(sys.argv[1])))
