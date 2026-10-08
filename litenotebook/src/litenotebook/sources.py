"""Source records and their lifecycle (no heavy imports: safe for page rendering).

Lifecycle of a source::

    upload/CLI ──register──▶ pending ──worker claims──▶ processing
                                                          │ extracting → chunking
                                                          │ → embedding → indexing
                                   ┌──────────────────────┴──────────┐
                                   ▼                                 ▼
                                 ready                             failed
                                                                     │ retry
                                                         pending ◀───┘

* Pages and chunks are written only in the final transaction that also sets
  ``ready``; a pending/processing/failed source never has any, so retrieval
  can never see a half-ingested document.
* ``processing`` sources cannot be deleted or retried (the worker owns them).
* On application start, sources left pending/processing by an interrupted run
  become ``failed`` (see ``recover_interrupted``); nothing is re-run on boot.
* Duplicates are detected per notebook (``UNIQUE(notebook_id, sha256)``): the
  same PDF may live in several notebooks, each with its own chunks/index.

Stored files live at ``<data>/notebooks/<notebook_id>/sources/<source_id>.pdf``.
The original filename is display metadata only and never touches the filesystem.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from litenotebook.config import Settings

log = logging.getLogger(__name__)

STATUSES = ("pending", "processing", "ready", "failed")
STAGES = (
    "uploaded",
    "extracting",
    "chunking",
    "loading_model",
    "embedding",
    "indexing",
    "ready",
    "failed",
)
STAGE_LABELS = {
    "uploaded": "Uploaded",
    "extracting": "Extracting",
    "chunking": "Chunking",
    "loading_model": "Loading embedding model",
    "embedding": "Embedding",
    "indexing": "Indexing",
    "ready": "Ready",
    "failed": "Failed",
}
INTERRUPTED_MESSAGE = "Ingestion interrupted by application restart. Retry the source."
MAX_FILENAME_LENGTH = 200
NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


class SourceError(ValueError):
    pass


class SourceNotFoundError(SourceError):
    pass


class SourceBusyError(SourceError):
    pass


@dataclass(frozen=True)
class Source:
    id: int
    notebook_id: int
    title: str
    original_filename: str
    stored_path: str
    mime: str
    sha256: str
    size_bytes: int
    page_count: int | None
    status: str
    stage: str
    progress_done: int | None
    progress_total: int | None
    ocr: bool
    error: str | None
    timings: dict[str, float]
    attempts: int
    created_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str
    chunk_count: int
    pages_ocr: int
    pages_empty: int

    @property
    def stage_label(self) -> str:
        return STAGE_LABELS.get(self.stage, self.stage)

    @property
    def active(self) -> bool:
        return self.status in ("pending", "processing")

    @property
    def file_type(self) -> str:
        return "PDF" if self.mime == "application/pdf" else self.mime

    @property
    def ingest_seconds(self) -> float | None:
        return self.timings.get("total")


def display_filename(name: str) -> str:
    """Sanitize a client-supplied filename for *display/metadata* only.

    Drops any directory part (both separators), control/format characters and
    surrounding dots/space, and normalizes Unicode. The result is never used to
    build a filesystem path; storage uses the integer source id.
    """
    name = unicodedata.normalize("NFC", name or "")
    name = name.replace("\\", "/")
    name = PurePath(name).name if "/" in name else name
    name = "".join(
        ch for ch in name if unicodedata.category(ch) not in ("Cc", "Cf", "Cs")
    )
    name = name.strip().strip(".").strip()
    if len(name) > MAX_FILENAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        name = (
            stem[: MAX_FILENAME_LENGTH - len(ext) - 1] + "." + ext
            if dot and len(ext) <= 10
            else name[:MAX_FILENAME_LENGTH]
        )
    return name or "document.pdf"


_SELECT = (
    "SELECT s.*, "
    "(SELECT count(*) FROM chunks c WHERE c.source_id = s.id) AS n_chunks, "
    "(SELECT count(*) FROM pages p WHERE p.source_id = s.id AND p.method = 'ocr') "
    "AS n_ocr, "
    "(SELECT count(*) FROM pages p WHERE p.source_id = s.id AND p.method = 'empty') "
    "AS n_empty "
    "FROM sources s"
)


def _from_row(row: sqlite3.Row) -> Source:
    try:
        timings: dict[str, float] = json.loads(row["timings"] or "{}")
    except (TypeError, ValueError):
        timings = {}
    return Source(
        id=row["id"],
        notebook_id=row["notebook_id"],
        title=row["title"],
        original_filename=row["original_filename"],
        stored_path=row["stored_path"],
        mime=row["mime"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        page_count=row["page_count"],
        status=row["status"],
        stage=row["stage"],
        progress_done=row["progress_done"],
        progress_total=row["progress_total"],
        ocr=bool(row["ocr"]),
        error=row["error"],
        timings=timings,
        attempts=row["attempts"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=row["updated_at"],
        chunk_count=row["n_chunks"],
        pages_ocr=row["n_ocr"],
        pages_empty=row["n_empty"],
    )


def list_sources(conn: sqlite3.Connection, notebook_id: int) -> list[Source]:
    rows = conn.execute(
        _SELECT + " WHERE s.notebook_id = ? ORDER BY s.created_at DESC, s.id DESC",
        (notebook_id,),
    ).fetchall()
    return [_from_row(r) for r in rows]


def get_source(conn: sqlite3.Connection, notebook_id: int, source_id: int) -> Source:
    """Notebook-scoped lookup: a source id from another notebook is 'not found'."""
    row = conn.execute(
        _SELECT + " WHERE s.id = ? AND s.notebook_id = ?", (source_id, notebook_id)
    ).fetchone()
    if row is None:
        raise SourceNotFoundError(f"Source {source_id} not found in this notebook")
    return _from_row(row)


def find_by_sha256(
    conn: sqlite3.Connection, notebook_id: int, sha256: str
) -> Source | None:
    row = conn.execute(
        _SELECT + " WHERE s.notebook_id = ? AND s.sha256 = ?", (notebook_id, sha256)
    ).fetchone()
    return _from_row(row) if row else None


def pending_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        "SELECT id FROM sources WHERE status = 'pending' ORDER BY id"
    ).fetchall()
    return [r[0] for r in rows]


def set_progress(
    conn: sqlite3.Connection,
    source_id: int,
    stage: str,
    done: int | None = None,
    total: int | None = None,
) -> None:
    with conn:
        conn.execute(
            f"UPDATE sources SET stage = ?, progress_done = ?, progress_total = ?, "
            f"updated_at = {NOW_SQL} WHERE id = ? AND status = 'processing'",
            (stage, done, total, source_id),
        )


def mark_failed(
    conn: sqlite3.Connection,
    source_id: int,
    message: str,
    timings: dict[str, Any] | None = None,
) -> None:
    """Fail a source and remove any derived rows (pages → chunks → FTS cascade)."""
    with conn:
        conn.execute("DELETE FROM pages WHERE source_id = ?", (source_id,))
        conn.execute(
            f"UPDATE sources SET status = 'failed', stage = 'failed', error = ?, "
            f"timings = coalesce(?, timings), finished_at = {NOW_SQL}, "
            f"updated_at = {NOW_SQL} WHERE id = ?",
            (message, json.dumps(timings) if timings else None, source_id),
        )


def reset_for_retry(
    conn: sqlite3.Connection, settings: Settings, notebook_id: int, source_id: int
) -> Source:
    """failed → pending, reusing the stored file and the same source record."""
    source = get_source(conn, notebook_id, source_id)
    if source.status != "failed":
        raise SourceBusyError(f"Only failed sources can be retried ({source.status})")
    if stored_file(settings, source) is None:
        raise SourceError(
            "The stored copy of this PDF is missing; delete the source and upload "
            "it again"
        )
    with conn:
        conn.execute("DELETE FROM pages WHERE source_id = ?", (source_id,))
        cur = conn.execute(
            f"UPDATE sources SET status = 'pending', stage = 'uploaded', error = NULL, "
            f"progress_done = NULL, progress_total = NULL, timings = NULL, "
            f"started_at = NULL, finished_at = NULL, updated_at = {NOW_SQL} "
            f"WHERE id = ? AND notebook_id = ? AND status = 'failed'",
            (source_id, notebook_id),
        )
        if cur.rowcount != 1:
            raise SourceBusyError("The source changed state; reload and try again")
    return get_source(conn, notebook_id, source_id)


def recover_interrupted(conn: sqlite3.Connection) -> list[int]:
    """Fail sources a previous run left pending/processing. Never re-runs them."""
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM sources WHERE status IN ('pending', 'processing')"
        )
    ]
    for source_id in ids:
        mark_failed(conn, source_id, INTERRUPTED_MESSAGE)
    if ids:
        log.warning("Marked %d interrupted source(s) as failed", len(ids))
    return ids


def stored_file(settings: Settings, source: Source) -> Path | None:
    """The source's file if it is a regular file inside its notebook's sources
    directory; None otherwise (missing, symlink, or a path that escapes)."""
    sources_dir = settings.sources_dir(source.notebook_id)
    candidate = settings.data_dir / source.stored_path
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        resolved = candidate.resolve(strict=True)
        root = sources_dir.resolve(strict=True)
    except OSError:
        return None
    if resolved.parent != root or not resolved.is_relative_to(
        settings.data_dir.resolve()
    ):
        return None
    return resolved


def remove_stored_file(settings: Settings, source: Source) -> bool:
    path = stored_file(settings, source)
    if path is None:
        return False
    try:
        path.unlink()
    except OSError as e:
        log.warning("Could not remove stored file of source %s: %s", source.id, e)
        return False
    return True


def delete_source_record(
    conn: sqlite3.Connection, notebook_id: int, source_id: int
) -> Source:
    """Delete the row (pages, chunks and FTS rows cascade). Callers then remove
    the file and sync FAISS; see ``ingest.pipeline.delete_source``."""
    source = get_source(conn, notebook_id, source_id)
    with conn:
        cur = conn.execute(
            "DELETE FROM sources WHERE id = ? AND notebook_id = ? "
            "AND status <> 'processing'",
            (source_id, notebook_id),
        )
    if cur.rowcount != 1:
        raise SourceBusyError(
            "This source is being ingested right now; wait for it to finish"
        )
    return source


def has_processing_sources(conn: sqlite3.Connection, notebook_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sources WHERE notebook_id = ? AND status = 'processing' LIMIT 1",
        (notebook_id,),
    ).fetchone()
    return row is not None
