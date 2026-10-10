"""Server-side page rendering with a derived, deletable on-disk cache.

Pages are rendered lazily with PyMuPDF (passive rendering: PDF JavaScript,
actions and links are never executed) into
``<data>/notebooks/<nb>/cache/pages/<source>/<page>-<scale>.png``. Paths are
built from integer ids only; the stored PDF is never modified and documents
are opened per request and closed again. Scales are quantised so the cache
holds a small, bounded set of variants.
"""

from __future__ import annotations

import contextlib
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from litenotebook.config import Settings
from litenotebook.platform_utils.filesystem import remove_tree, replace_file
from litenotebook.sources import Source, get_source, stored_file

SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
MAX_PIXELS_SIDE = 4000


class ReaderError(ValueError):
    pass


class PageNotFoundError(ReaderError):
    pass


@dataclass(frozen=True)
class ReaderSource:
    source: Source
    path: Path
    page_count: int


class RenderCacheUnavailable(Exception):  # noqa: N818 - carries the result
    """The page rendered but could not be cached; ``png`` holds the image."""

    def __init__(self, png: bytes) -> None:
        super().__init__("The page cache folder is not writable")
        self.png = png


def nearest_scale(value: float) -> float:
    return min(SCALES, key=lambda s: (abs(s - value), s))


def open_source(
    conn: sqlite3.Connection, settings: Settings, notebook_id: int, source_id: int
) -> ReaderSource:
    """Notebook-scoped source with its stored PDF (never a client path)."""
    source = get_source(conn, notebook_id, source_id)  # SourceNotFoundError
    path = stored_file(settings, source)
    if path is None:
        raise ReaderError("The PDF file of this source is not available")
    page_count = source.page_count
    if not page_count:
        import pymupdf

        from litenotebook.ingest.pdflock import PDF_LOCK

        with PDF_LOCK, pymupdf.open(path, filetype="pdf") as doc:
            page_count = int(doc.page_count)
    return ReaderSource(source, path, int(page_count))


def check_page(reader: ReaderSource, page_no: int) -> int:
    if not 1 <= int(page_no) <= reader.page_count:
        raise PageNotFoundError(
            f"Page {page_no} does not exist (this PDF has {reader.page_count} pages)"
        )
    return int(page_no)


def cache_root(settings: Settings, notebook_id: int) -> Path:
    return settings.notebook_dir(int(notebook_id)) / "cache" / "pages"


def cache_dir(settings: Settings, notebook_id: int, source_id: int) -> Path:
    return cache_root(settings, notebook_id) / str(int(source_id))


def render_page(
    settings: Settings, reader: ReaderSource, page_no: int, scale: float
) -> tuple[Path, bool]:
    """Path of the cached PNG for this page/scale; renders it on a miss.
    Returns (path, was_cached)."""
    page_no = check_page(reader, page_no)
    scale = nearest_scale(scale)
    target = (
        cache_dir(settings, reader.source.notebook_id, reader.source.id)
        / f"{page_no}-{int(scale * 100)}.png"
    )
    if target.is_file():
        return target, True
    import pymupdf

    from litenotebook.ingest.pdflock import PDF_LOCK

    with PDF_LOCK, pymupdf.open(reader.path, filetype="pdf") as doc:
        page = doc[page_no - 1]
        longest = max(page.rect.width, page.rect.height) or 1.0
        effective = min(scale, MAX_PIXELS_SIDE / longest)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(effective, effective), alpha=False)
        data = pix.tobytes("png")
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_bytes(data)
            try:
                replace_file(tmp, target)
            except PermissionError:
                # Windows: another request is serving the same page right now
                # and holds it open; that render is identical, so keep it.
                if not target.is_file():
                    raise
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
    except OSError as e:
        # The cache is only an optimisation: an unwritable or full cache
        # folder still lets the Reader show the page.
        raise RenderCacheUnavailable(data) from e
    return target, False


def remove_cache(settings: Settings, notebook_id: int, source_id: int) -> bool:
    """Delete one source's render cache — only inside the notebook's cache dir."""
    root = cache_root(settings, notebook_id)
    target = cache_dir(settings, notebook_id, source_id)
    if not target.exists() or target.is_symlink():
        with contextlib.suppress(OSError):
            if target.is_symlink():
                target.unlink()
        return False
    try:
        resolved, base = target.resolve(strict=True), root.resolve(strict=True)
    except OSError:
        return False
    if resolved.parent != base or not resolved.is_relative_to(
        settings.data_dir.resolve()
    ):
        return False
    # Derived data: on Windows a page being served at this moment cannot be
    # deleted; anything left over is harmless and removed with its notebook.
    with contextlib.suppress(OSError):
        remove_tree(resolved)
    return True


def cache_size(settings: Settings, notebook_id: int, source_id: int) -> int:
    directory = cache_dir(settings, notebook_id, source_id)
    if not directory.is_dir():
        return 0
    return sum(f.stat().st_size for f in directory.glob("*.png") if f.is_file())
