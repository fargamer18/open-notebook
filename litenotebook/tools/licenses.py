"""Dependency and license inventory for the desktop packages.

    python tools/licenses.py [OUT.md]

Three groups, each with name, version, license and source:

* **Bundled** — every Python distribution in the core app or in the optional
  embedding runtime (the dependency closure of their direct requirements,
  from installed package metadata), and the native components inside them;
* **Optional, downloaded** — what LiteNotebook can download when the user
  asks (llama.cpp, models, MiniLM, speech engine and voices), with the
  pinned versions from the catalogs; never bundled;
* **Development only** — test, lint and build tools; never shipped.

License texts are not pasted: each package's own license file is inside its
installed distribution and its source link is listed. Not legal advice.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import re
import sys
from pathlib import Path

CATALOG = (
    Path(__file__).resolve().parents[1] / "src" / "litenotebook" / "ai" / "catalog"
)

CORE_ROOTS = [
    "fastapi",
    "uvicorn",
    "jinja2",
    "python-multipart",
    "httpx",
    "pymupdf",
    "numpy",
    "faiss-cpu",
    "markdown-it-py",
    "nh3",
    "tokenizers",
    "huggingface-hub",
]
WORKER_ROOTS = [
    "sentence-transformers",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "numpy",
]
DEV_ROOTS = ["pytest", "ruff", "mypy", "pyinstaller", "pyinstaller-hooks-contrib"]
_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")

NON_PYTHON = [
    (
        "CPython runtime",
        "bundled (core, runtime)",
        "PSF-2.0",
        "https://www.python.org/",
    ),
    ("SQLite (in CPython)", "bundled (core)", "Public domain", "https://sqlite.org/"),
    (
        "MuPDF (inside PyMuPDF)",
        "bundled (core)",
        "AGPL-3.0-or-later, or Artifex commercial",
        "https://mupdf.com/",
    ),
    (
        "FAISS (inside faiss-cpu)",
        "bundled (core)",
        "MIT",
        "https://github.com/facebookresearch/faiss",
    ),
    (
        "OpenBLAS / libgfortran (numpy, faiss, scipy wheels)",
        "bundled",
        "BSD-3-Clause / GPL-3.0 with GCC runtime exception",
        "https://www.openblas.net/",
    ),
    (
        "libstdc++ / libgcc_s (Linux)",
        "bundled",
        "GPL-3.0 with GCC runtime library exception",
        "https://gcc.gnu.org/",
    ),
    (
        "PyInstaller bootloader",
        "bundled (both)",
        "GPL-2.0-or-later with bootloader exception",
        "https://pyinstaller.org/",
    ),
    ("htmx (vendored web asset)", "bundled (core)", "0BSD", "https://htmx.org/"),
    (
        "PyTorch native libraries (libtorch, libgomp)",
        "bundled (runtime only)",
        "BSD-3-Clause; libgomp GPL-3.0 with GCC runtime exception",
        "https://pytorch.org/",
    ),
]


def optional_components() -> list[tuple[str, str, str, str]]:
    """(name, version, license, source) of downloadable optional parts,
    read from the pinned catalogs."""
    rows: list[tuple[str, str, str, str]] = []
    runtimes = json.loads((CATALOG / "runtimes.json").read_text("utf-8"))["runtimes"]
    seen = set()
    for r in runtimes:
        key = (r["project"], r["version"])
        if key not in seen:
            seen.add(key)
            rows.append(
                (
                    r["project"] + " (llama-server)",
                    r["version"],
                    r["license"],
                    r["source_url"],
                )
            )
    models = json.loads((CATALOG / "models.json").read_text("utf-8"))
    for m in models["models"] if isinstance(models, dict) else models:
        rows.append(
            (
                m.get("display_name") or m["id"],
                m["id"],
                m.get("license", "?"),
                m.get("model_page") or f"https://huggingface.co/{m.get('repository')}",
            )
        )
    emb = json.loads((CATALOG / "embeddings.json").read_text("utf-8"))
    for e in emb["embeddings"] if isinstance(emb, dict) else emb:
        rows.append(
            (
                e["display_name"],
                "commit " + e["revision"][:12],
                e["license"],
                e["model_page"],
            )
        )
    speech = json.loads((CATALOG / "speech.json").read_text("utf-8"))
    engine = speech["engines"][0]
    rows.append(
        (
            engine["project"] + " speech engine",
            engine["version"],
            engine["license"],
            engine["source_url"],
        )
    )
    rows.append(
        (
            "espeak-ng / espeak-ng-data (inside the speech downloads)",
            "as shipped",
            "GPL-3.0",
            "https://github.com/espeak-ng/espeak-ng",
        )
    )
    for v in speech.get("voices", []):
        rows.append(
            (
                v.get("display_name") or v["id"],
                v["id"],
                v.get("license", "?"),
                v.get("model_card_url") or v.get("model_page") or "",
            )
        )
    return rows


def _source(dist: md.Distribution) -> str:
    meta = dist.metadata
    urls = {}
    for raw in meta.get_all("Project-URL") or []:
        label, _, url = raw.partition(",")
        urls[label.strip().lower()] = url.strip()
    for key in ("source", "source code", "repository", "homepage", "home"):
        if urls.get(key):
            return urls[key]
    return meta.get("Home-page") or next(iter(urls.values()), "")


def _license(dist: md.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::") and "OSI Approved" != c.split("::")[-1].strip()
    ]
    text = (meta.get("License") or "").strip()
    if classifiers:
        return "; ".join(dict.fromkeys(classifiers))
    if text and len(text) < 120:
        return text
    return text.splitlines()[0][:100] + " (see package)" if text else "UNKNOWN"


def closure(roots: list[str]) -> dict[str, md.Distribution]:
    seen: dict[str, md.Distribution] = {}
    todo = list(roots)
    while todo:
        name = todo.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        seen[key] = dist
        for req in dist.requires or []:
            if "extra ==" in req:
                continue
            if (
                ";" in req
                and "sys_platform" in req
                and "win32" in req
                and sys.platform != "win32"
            ):
                continue
            m = _NAME.match(req.strip())
            if m:
                todo.append(m.group(1))
    return seen


def write_inventory(out: Path) -> list[tuple[str, str, str, str]]:
    core = closure(CORE_ROOTS)
    worker = closure(WORKER_ROOTS)
    shipped = set(core) | set(worker)
    dev = {k: v for k, v in closure(DEV_ROOTS).items() if k not in shipped}
    rows = []
    for key in sorted(shipped):
        dist = core.get(key) or worker[key]
        where = " + ".join(
            w for w, s in (("core", core), ("embedding runtime", worker)) if key in s
        )
        rows.append((dist.metadata["Name"], dist.version, where, _license(dist)))
    sources = {
        (core.get(k) or worker[k]).metadata["Name"]: _source(core.get(k) or worker[k])
        for k in shipped
    }
    lines = [
        "# LiteNotebook — third-party components and licenses",
        "",
        "Generated by `tools/licenses.py` from installed package metadata and the",
        "pinned download catalogs. An inventory for review, **not legal advice**.",
        "License texts are not repeated here: each package ships its own license",
        "file inside the bundled distribution, and its source is linked.",
        "",
        "> **PyMuPDF / MuPDF is AGPL-3.0-or-later (or a commercial license from",
        "> Artifex)** and is bundled in the core app. This is an open release",
        "> blocker — see docs/pdf-licensing-decision.md.",
        "",
        "## 1. Bundled — Python packages",
        "",
        "In the core app (always) or in the optional embedding runtime (a separate",
        "download that contains PyTorch).",
        "",
        "| Package | Version | In | License (from metadata) | Source |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {n} | {v} | {w} | {lic} | {sources.get(n, '')} |" for n, v, w, lic in rows
    ]
    lines += [
        "",
        "## 2. Bundled — native and other components",
        "",
        "| Component | Status | License | Source |",
        "|---|---|---|---|",
    ]
    lines += [f"| {n} | {s} | {lic} | {src} |" for n, s, lic, src in NON_PYTHON]
    lines += [
        "",
        "## 3. Optional, downloaded on request (never bundled)",
        "",
        "Installed only when the user chooses them in the setup wizard or",
        "Settings → Local AI, from the pinned, checksum-verified catalog entries.",
        "",
        "| Component | Version / id | License | Source |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {n} | {v} | {lic} | {src} |" for n, v, lic, src in optional_components()
    ]
    lines += [
        "",
        "## 4. Development only (never shipped)",
        "",
        "| Package | Version | License | Source |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {d.metadata['Name']} | {d.version} | {_license(d)} | {_source(d)} |"
        for _, d in sorted(dev.items())
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("THIRD_PARTY_LICENSES.md")
    print(f"{len(write_inventory(target))} packages → {target}")
