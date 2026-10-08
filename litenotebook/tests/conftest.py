import os
import re
import sys
import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from litenotebook.config import EMBED_DIM, EMBED_MODEL, Settings, load_settings

# Tests must never reach the network for model downloads.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(EMBED_MODEL, "model.safetensors"), str)


def pytest_configure(config):  # type: ignore[no-untyped-def]
    config.addinivalue_line(
        "markers",
        "posix_only: relies on POSIX semantics (permission bits, signals, /proc); "
        "skipped on Windows",
    )


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    run_llm = os.environ.get("LITENOTEBOOK_TEST_LLM") == "1"
    have_model = _minilm_cached()
    for item in items:
        if "posix_only" in item.keywords and sys.platform == "win32":
            item.add_marker(pytest.mark.skip(reason="POSIX-specific behaviour"))
        if "llm" in item.keywords and not run_llm:
            item.add_marker(pytest.mark.skip(reason="set LITENOTEBOOK_TEST_LLM=1"))
        if "model" in item.keywords and not have_model:
            item.add_marker(pytest.mark.skip(reason="MiniLM not in the HF cache"))


def symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    """Symlinks need Developer Mode or admin rights on Windows."""
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"cannot create symlinks here: {e}")


def process_gone(pid: int, timeout: float = 10.0) -> bool:
    """True once ``pid`` no longer exists (and, on Linux, is not a zombie)."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return False


def _pid_alive(pid: int) -> bool:
    if sys.platform.startswith("linux"):
        # running, sleeping — or a zombie ("Z"), which still counts as a leak
        return Path(f"/proc/{pid}/stat").exists()
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX only: signal 0 just checks existence
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors: fast, no model download.

    Texts sharing words get similar vectors, so semantic search is meaningful.
    It claims MiniLM's name/dimension so notebooks accept it in tests.
    """

    model_name = EMBED_MODEL
    dim = EMBED_DIM

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                h = zlib.crc32(word.encode())
                out[row, h % EMBED_DIM] += 1.0 if (h >> 16) & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm == 0:
                out[row, 0], norm = 1.0, 1.0
            out[row] /= norm
        return out


def make_pdf(
    path: Path, pages: list[str], *, image_only: set[int] | None = None
) -> Path:
    """Write a PDF with one text page per entry (1-based numbers in image_only
    become scanned-style image pages with no text layer)."""
    image_only = image_only or set()
    doc = pymupdf.open()
    for number, text in enumerate(pages, 1):
        page = doc.new_page(width=595, height=842)
        rect = pymupdf.Rect(40, 40, 555, 802)
        if number in image_only:
            scratch = pymupdf.open()
            src = scratch.new_page(width=595, height=842)
            src.insert_textbox(rect, text, fontsize=14)
            pix = src.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            page.insert_image(page.rect, pixmap=pix)
            scratch.close()
        elif text:
            assert page.insert_textbox(rect, text, fontsize=8) >= 0, "text overflow"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings rooted in a throwaway data directory."""
    return load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})


@pytest.fixture
def conn(settings: Settings) -> Iterator:  # type: ignore[type-arg]
    from litenotebook.db import open_db

    connection = open_db(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(conn, settings):  # type: ignore[no-untyped-def]
    from litenotebook.index.faiss_store import FaissStore

    return FaissStore(conn, settings)


@pytest.fixture
def notebook(conn):  # type: ignore[no-untyped-def]
    from litenotebook.notebooks import create_notebook

    return create_notebook(conn, "demo")
