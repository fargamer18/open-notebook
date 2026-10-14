# PyInstaller spec: the LiteNotebook core app (one folder, no torch).
# Built by tools/build.py; options come from the environment it sets.
import os
import sys

sys.path.insert(0, SPECPATH)
from _binaries import filter_binaries  # noqa: E402
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.environ["LN_ROOT"]
ICON = os.environ.get("LN_ICON") or None
NAME = os.environ.get("LN_APP_NAME", "litenotebook")

# Never in the core app: the embedding stack lives in the separate worker.
HEAVY = [
    "torch", "torchvision", "torchaudio", "sentence_transformers", "transformers",
    "scipy", "sklearn", "sympy", "networkx", "safetensors", "accelerate",
    "tkinter", "matplotlib", "IPython", "pytest", "mypy", "ruff", "PyInstaller",
    "onnxruntime", "sherpa_onnx", "pandas",
    # PDFs are read with PDFium (pypdfium2, collected by its PyInstaller
    # hooks); PyMuPDF must never be bundled even if a build machine has it.
    "pymupdf", "fitz",
]

a = Analysis(
    [os.path.join(ROOT, "packaging", "pyinstaller", "app_entry.py")],
    pathex=[os.path.join(ROOT, "src")],
    datas=collect_data_files("litenotebook", excludes=["**/__pycache__/**"]),
    hiddenimports=(
        collect_submodules("litenotebook", filter=lambda m: ".embed.worker" not in m)
        + collect_submodules("uvicorn")
        + collect_submodules("faiss")
        + ["multipart", "python_multipart"]
    ),
    excludes=HEAVY,
    noarchive=False,
)
a.binaries = filter_binaries(a.binaries)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=os.environ.get("LN_CONSOLE", "0") == "1",  # Windows: no console window
    icon=ICON,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=NAME)
