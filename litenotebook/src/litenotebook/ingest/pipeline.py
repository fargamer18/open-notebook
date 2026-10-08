"""PDF ingestion: register (validate + store) → process (extract → chunk → embed
→ SQLite in one transaction → FAISS sync). See ``litenotebook.sources`` for the
source lifecycle.

``ingest_pdf`` keeps the Phase 1 all-or-nothing behaviour for the CLI; the web
app registers a source on upload and lets the background worker process it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from litenotebook.config import Settings
from litenotebook.embed.embedder import (
    EmbeddingError,
    EmbeddingModelUnavailableError,
    TextEmbedder,
    to_blob,
)
from litenotebook.ingest.chunker import ChunkingError, chunk_pages
from litenotebook.ingest.pdf import PdfError, extract_pages, has_pdf_header, inspect_pdf
from litenotebook.notebooks import Notebook, get_notebook
from litenotebook.sources import (
    NOW_SQL,
    Source,
    SourceError,
    delete_source_record,
    display_filename,
    find_by_sha256,
    get_source,
    mark_failed,
    remove_stored_file,
    set_progress,
    stored_file,
)

if TYPE_CHECKING:  # FAISS is imported only by callers that actually index
    from litenotebook.index.faiss_store import FaissStore, SyncResult

log = logging.getLogger(__name__)

PDF_MIME = "application/pdf"
EMBED_BATCH = 64  # chunks per encode() call; also the embedding progress step
PROGRESS_MIN_INTERVAL_S = 0.5  # throttle per-page progress writes

ProgressHook = Callable[[str, int | None, int | None], None]


class IngestError(ValueError):
    pass


class UnsupportedFileError(IngestError):
    """The file is not a PDF (content sniffed; name and MIME type are ignored)."""


class DuplicateSourceError(IngestError):
    def __init__(self, existing: Source, filename: str) -> None:
        state = "" if existing.status == "ready" else f" ({existing.status})"
        super().__init__(
            f"{filename} is already in this notebook as "
            f"“{existing.original_filename}”{state}"
        )
        self.source_id = existing.id
        self.existing = existing


@dataclass
class IngestReport:
    source_id: int
    filename: str
    pages: int
    pages_ocr: int
    pages_empty: int
    chunks: int
    faiss: SyncResult
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_embedder(notebook: Notebook, embedder: TextEmbedder) -> None:
    if (
        embedder.model_name != notebook.embed_model
        or embedder.dim != notebook.embed_dim
    ):
        raise IngestError(
            f"Notebook expects {notebook.embed_model} ({notebook.embed_dim}-d) "
            f"embeddings, embedder is {embedder.model_name} ({embedder.dim}-d)"
        )


def _place(src: Path, dest: Path, *, move: bool) -> None:
    """Atomically put ``src`` at ``dest`` (a rename when moving on one filesystem)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        if move:
            try:
                src.replace(tmp)
            except OSError:  # different filesystem
                shutil.copyfile(src, tmp)
                src.unlink()
        else:
            shutil.copyfile(src, tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def register_source(
    conn: sqlite3.Connection,
    settings: Settings,
    notebook: Notebook,
    path: Path,
    *,
    filename: str | None = None,
    ocr: bool = False,
    move: bool = False,
) -> Source:
    """Validate a PDF, store it under the notebook and record it as ``pending``.

    ``filename`` is the user-facing name (metadata only); the file is stored as
    ``<source_id>.pdf``. With ``move=True`` the input file is consumed.
    """
    name = display_filename(filename if filename is not None else path.name)
    if not path.is_file():
        raise IngestError(f"File not found: {name}")
    if not has_pdf_header(path):
        raise UnsupportedFileError(
            f"{name} is not a PDF. Only PDF files are supported."
        )
    try:
        page_count = inspect_pdf(path, name=name)
    except PdfError as e:
        raise IngestError(str(e)) from e

    sha256 = file_sha256(path)
    existing = find_by_sha256(conn, notebook.id, sha256)
    if existing is not None:
        raise DuplicateSourceError(existing, name)

    size = path.stat().st_size
    dest: Path | None = None
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO sources (notebook_id, title, original_filename, "
                "stored_path, mime, sha256, size_bytes, page_count, status, stage, "
                "ocr) VALUES (?, ?, ?, '', ?, ?, ?, ?, 'pending', 'uploaded', ?)",
                (
                    notebook.id,
                    Path(name).stem or name,
                    name,
                    PDF_MIME,
                    sha256,
                    size,
                    page_count,
                    int(ocr),
                ),
            )
            source_id = cur.lastrowid
            assert source_id is not None
            dest = settings.source_file(notebook.id, source_id)
            conn.execute(
                "UPDATE sources SET stored_path = ? WHERE id = ?",
                (str(dest.relative_to(settings.data_dir)), source_id),
            )
            _place(path, dest, move=move)
    except sqlite3.IntegrityError as e:  # lost a race with an identical upload
        if dest is not None:
            dest.unlink(missing_ok=True)
        existing = find_by_sha256(conn, notebook.id, sha256)
        if existing is not None:
            raise DuplicateSourceError(existing, name) from e
        raise
    except BaseException:
        if dest is not None:
            dest.unlink(missing_ok=True)
        raise
    return get_source(conn, notebook.id, source_id)


class _Progress:
    """Writes stage/progress to SQLite for the polling UI (throttled)."""

    def __init__(
        self, conn: sqlite3.Connection, source_id: int, hook: ProgressHook | None
    ) -> None:
        self.conn, self.source_id, self.hook = conn, source_id, hook
        self.stage = ""
        self._last = 0.0

    def __call__(
        self, stage: str, done: int | None = None, total: int | None = None
    ) -> None:
        now = time.monotonic()
        final = done is not None and done == total
        if (
            stage == self.stage
            and not final
            and now - self._last < PROGRESS_MIN_INTERVAL_S
        ):
            return
        self.stage, self._last = stage, now
        set_progress(self.conn, self.source_id, stage, done, total)
        if self.hook is not None:
            self.hook(stage, done, total)


def _claim(conn: sqlite3.Connection, source_id: int) -> bool:
    with conn:
        cur = conn.execute(
            f"UPDATE sources SET status = 'processing', stage = 'extracting', "
            f"error = NULL, timings = NULL, progress_done = NULL, "
            f"progress_total = NULL, attempts = attempts + 1, "
            f"started_at = {NOW_SQL}, finished_at = NULL, updated_at = {NOW_SQL} "
            f"WHERE id = ? AND status = 'pending'",
            (source_id,),
        )
    return cur.rowcount == 1


def process_source(
    conn: sqlite3.Connection,
    settings: Settings,
    source_id: int,
    embedder: TextEmbedder,
    store: FaissStore,
    *,
    on_progress: ProgressHook | None = None,
) -> IngestReport:
    """Ingest a ``pending`` source end to end.

    On any failure the source becomes ``failed`` with a user-facing message (no
    paths, no tracebacks), no pages/chunks remain, FAISS is re-synced if it was
    touched, and ``IngestError`` is raised. The stored file is kept for retry.
    """
    row = conn.execute(
        "SELECT notebook_id FROM sources WHERE id = ?", (source_id,)
    ).fetchone()
    if row is None:
        raise IngestError(f"Source {source_id} no longer exists")
    if not _claim(conn, source_id):
        raise IngestError(f"Source {source_id} is not pending")

    started = time.perf_counter()
    timings: dict[str, float] = {}
    stage = "extracting"
    progress = _Progress(conn, source_id, on_progress)
    committed = False

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = round(now - since, 4)
        return now

    try:
        source = get_source(conn, row["notebook_id"], source_id)
        _check_embedder(get_notebook(conn, source.notebook_id), embedder)
        path = stored_file(settings, source)
        if path is None:
            raise IngestError(
                "The stored copy of this PDF is missing; delete the source and "
                "upload it again"
            )
        name = source.original_filename

        t = time.perf_counter()
        progress("extracting", 0, source.page_count)
        pages = extract_pages(
            path,
            ocr=source.ocr,
            name=name,
            on_page=lambda done, total: progress("extracting", done, total),
        )
        t = lap("extract", t)

        stage = "chunking"
        progress("chunking")
        chunks = chunk_pages(
            ((p.page_no, p.text) for p in pages),
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        t = lap("chunk", t)
        if not chunks:
            raise IngestError(
                "No extractable text"
                + (
                    " even with OCR"
                    if source.ocr
                    else " — this looks like a scanned PDF. Delete it and upload "
                    "it again with “OCR pages with no text layer” ticked"
                )
            )

        stage = "embedding"
        load = getattr(embedder, "load", None)
        if callable(load):  # report model load time separately from embedding
            if not getattr(embedder, "loaded", True):
                progress("loading_model")  # a cold worker: no fake numbers
            load()
            t = lap("model_load", t)
        progress("embedding", 0, len(chunks))
        blobs: list[bytes] = []
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            vectors = embedder.encode([c.text for c in batch])
            if len(vectors) != len(batch):
                raise IngestError(
                    f"Got {len(vectors)} embeddings for {len(batch)} chunks"
                )
            blobs.extend(to_blob(v) for v in vectors)
            progress("embedding", len(blobs), len(chunks))
        t = lap("embed", t)

        stage = "indexing"
        progress("indexing")
        with conn:
            conn.execute("DELETE FROM pages WHERE source_id = ?", (source_id,))
            page_ids: dict[int, int] = {}
            for page in pages:
                page_cur = conn.execute(
                    "INSERT INTO pages (source_id, page_no, text, method) "
                    "VALUES (?, ?, ?, ?)",
                    (source_id, page.page_no, page.text, page.method),
                )
                assert page_cur.lastrowid is not None
                page_ids[page.page_no] = page_cur.lastrowid
            conn.executemany(
                "INSERT INTO chunks (notebook_id, source_id, page_id, page_no, ord, "
                "char_start, char_end, heading, text, n_tokens, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        source.notebook_id,
                        source_id,
                        page_ids[c.page_no],
                        c.page_no,
                        c.ord,
                        c.char_start,
                        c.char_end,
                        c.heading,
                        c.text,
                        c.n_tokens,
                        blob,
                    )
                    for c, blob in zip(chunks, blobs, strict=True)
                ],
            )
            # timings so far go in with "ready" (FAISS time is added below), so
            # a Ready source is never shown without them
            timings["total"] = round(time.perf_counter() - started, 4)
            cur = conn.execute(
                f"UPDATE sources SET status = 'ready', stage = 'ready', "
                f"page_count = ?, progress_done = NULL, progress_total = NULL, "
                f"timings = ?, finished_at = {NOW_SQL}, updated_at = {NOW_SQL} "
                f"WHERE id = ? AND status = 'processing'",
                (len(pages), json.dumps(timings), source_id),
            )
            if cur.rowcount != 1:  # deleted or recovered underneath us
                raise IngestError("The source was removed during ingestion")
        committed = True
        t = lap("sqlite", t)

        faiss_result = store.sync(source.notebook_id)
        lap("faiss", t)
        timings["total"] = round(time.perf_counter() - started, 4)
        with conn:
            conn.execute(
                "UPDATE sources SET timings = ? WHERE id = ?",
                (json.dumps(timings), source_id),
            )
    except Exception as e:
        timings["total"] = round(time.perf_counter() - started, 4)
        message = _user_message(e, stage)
        if isinstance(e, IngestError | PdfError | ChunkingError | EmbeddingError):
            log.warning("Ingestion of source %s failed: %s", source_id, message)
        else:
            log.exception("Ingestion of source %s failed while %s", source_id, stage)
        _fail(conn, store, source_id, message, timings, resync=committed)
        raise IngestError(message) from e

    empty = [p.page_no for p in pages if p.method == "empty"]
    warnings: list[str] = []
    if empty:
        warnings.append(
            f"{len(empty)} page(s) had no extractable text"
            + ("" if source.ocr else " (try --ocr for scanned pages)")
        )
    return IngestReport(
        source_id=source_id,
        filename=name,
        pages=len(pages),
        pages_ocr=sum(p.method == "ocr" for p in pages),
        pages_empty=len(empty),
        chunks=len(chunks),
        faiss=faiss_result,
        timings=timings,
        warnings=warnings,
    )


def _user_message(error: Exception, stage: str) -> str:
    """A message safe to show in the browser: no paths, no tracebacks."""
    if isinstance(error, IngestError | PdfError | ChunkingError):
        return str(error)
    if isinstance(error, EmbeddingModelUnavailableError):
        return str(error)
    if isinstance(error, EmbeddingError):
        return f"Embedding failed: {error}"
    if isinstance(error, MemoryError):
        return f"Ran out of memory while {stage}; close other programs and retry"
    if isinstance(error, sqlite3.OperationalError) and "locked" in str(error):
        return "The database was busy; retry the source"
    if isinstance(error, sqlite3.Error):  # SQLite messages never include paths
        return f"Database error while {stage}: {error}"
    return f"Unexpected error while {stage}; details are in the server log"


def _fail(
    conn: sqlite3.Connection,
    store: FaissStore,
    source_id: int,
    message: str,
    timings: dict[str, float],
    *,
    resync: bool,
) -> None:
    """Best effort: mark failed, drop derived rows, re-sync FAISS if it changed."""
    try:
        conn.rollback()
        row = conn.execute(
            "SELECT notebook_id FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        if row is None:
            return
        mark_failed(conn, source_id, message, timings)
        if resync:
            store.sync(row["notebook_id"])
    except Exception:  # FAISS self-repairs from SQLite on its next load()
        log.exception("Cleanup after failed ingestion of source %s failed", source_id)


def delete_source(
    conn: sqlite3.Connection,
    settings: Settings,
    notebook_id: int,
    source_id: int,
    store: FaissStore,
) -> Source:
    """Delete a source: row (pages/chunks/FTS cascade), stored file, FAISS vectors."""
    from litenotebook.reader.render import remove_cache

    source = delete_source_record(conn, notebook_id, source_id)  # annotations cascade
    remove_stored_file(settings, source)
    remove_cache(settings, notebook_id, source_id)  # derived page renders
    store.sync(notebook_id)
    return source


def ingest_pdf(
    conn: sqlite3.Connection,
    settings: Settings,
    notebook: Notebook,
    path: Path,
    embedder: TextEmbedder,
    store: FaissStore,
    *,
    ocr: bool = False,
) -> IngestReport:
    """Synchronous all-or-nothing ingestion (CLI): on failure nothing remains."""
    _check_embedder(notebook, embedder)
    source = register_source(conn, settings, notebook, path, ocr=ocr)
    try:
        return process_source(conn, settings, source.id, embedder, store)
    except BaseException:
        try:
            delete_source(conn, settings, notebook.id, source.id, store)
        except SourceError:
            pass
        raise
