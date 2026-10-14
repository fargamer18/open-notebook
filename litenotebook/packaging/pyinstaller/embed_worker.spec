# PyInstaller spec: the optional embedding worker (torch CPU + MiniLM stack).
# A console-subsystem program on Windows: LiteNotebook starts it with
# CREATE_NO_WINDOW, so no window appears, and its stdin/stdout pipes work.
import os
import sys

sys.path.insert(0, SPECPATH)
from _binaries import filter_binaries  # noqa: E402
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

ROOT = os.environ["LN_ROOT"]
NAME = "litenotebook-embed-worker"

datas = []
for dist in ("torch", "transformers", "sentence-transformers", "tokenizers",
             "huggingface-hub", "safetensors", "tqdm", "regex", "requests",
             "packaging", "filelock", "numpy", "pyyaml"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass
datas += collect_data_files("sentence_transformers")

a = Analysis(
    [os.path.join(ROOT, "packaging", "pyinstaller", "worker_entry.py")],
    pathex=[os.path.join(ROOT, "src")],
    datas=datas,
    hiddenimports=["litenotebook.embed.worker", "litenotebook.embed.protocol"],
    excludes=[
        "fastapi", "starlette", "uvicorn", "jinja2", "pymupdf", "fitz", "faiss",
        "pypdfium2", "pypdfium2_raw", "pypdfium2_cfg",
        "multipart", "nh3", "markdown_it", "tkinter", "matplotlib",
        "IPython", "pytest", "mypy", "PyInstaller", "pandas",
        "torchvision", "torchaudio",
    ],
    noarchive=False,
)
a.binaries = filter_binaries(a.binaries)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=NAME)
