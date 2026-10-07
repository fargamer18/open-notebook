"""PDF ingestion: extract → chunk → embed → SQLite (one transaction) → FAISS sync."""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from litenotebook.config import Settings
from litenotebook.embed.embedder import EmbeddingError, TextEmbedder, to_blob
from litenotebook.index.faiss_store import FaissStore, SyncResult
from litenotebook.ingest.chunker import chunk_pages
from litenotebook.ingest.pdf import PdfError, extract_pages
from litenotebook.notebooks import Notebook


class IngestError(ValueError):
    pass


class DuplicateSourceError(IngestError):
    def __init__(self, source_id: int, filename: str) -> None:
        super().__init__(f"{filename} is already in this notebook (source {source_id})")
        self.source_id = source_id


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


def safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\- ]+", "_", base).strip(" .") or "document"
    stem, suffix = Path(cleaned).stem[:100], Path(cleaned).suffix[:10]
    return f"{stem}{suffix}"


def _looks_like_pdf(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(1024).lstrip().startswith(b"%PDF-")


def _store_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(src, tmp)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


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
    started = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = now - since
        return now

    if not path.is_file():
        raise IngestError(f"File not found: {path}")
    if not _looks_like_pdf(path):
        raise IngestError(f"{path.name} is not a PDF (missing %PDF- header)")
    if (
        embedder.model_name != notebook.embed_model
        or embedder.dim != notebook.embed_dim
    ):
        raise IngestError(
            f"Notebook expects {notebook.embed_model} ({notebook.embed_dim}-d) "
            f"embeddings, embedder is {embedder.model_name} ({embedder.dim}-d)"
        )

    sha256 = file_sha256(path)
    existing = conn.execute(
        "SELECT id FROM sources WHERE notebook_id = ? AND sha256 = ?",
        (notebook.id, sha256),
    ).fetchone()
    if existing:
        raise DuplicateSourceError(existing["id"], path.name)

    t = time.perf_counter()
    try:
        pages = extract_pages(path, ocr=ocr)
    except PdfError as e:
        raise IngestError(str(e)) from e
    t = lap("extract", t)

    chunks = chunk_pages(
        ((p.page_no, p.text) for p in pages),
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    t = lap("chunk", t)

    try:
        vectors = embedder.encode([c.text for c in chunks])
        blobs = [to_blob(v) for v in vectors]
    except EmbeddingError as e:
        raise IngestError(f"Embedding failed: {e}") from e
    if len(blobs) != len(chunks):
        raise IngestError(f"Got {len(blobs)} embeddings for {len(chunks)} chunks")
    t = lap("embed", t)

    warnings: list[str] = []
    empty = [p.page_no for p in pages if p.method == "empty"]
    if empty:
        warnings.append(
            f"{len(empty)} page(s) had no extractable text"
            + ("" if ocr else " (try --ocr for scanned pages)")
        )
    status, error = "ready", None
    if not chunks:
        status, error = "error", "No extractable text"

    stored_rel = (
        Path("notebooks")
        / str(notebook.id)
        / "sources"
        / f"{sha256[:16]}-{safe_filename(path.name)}"
    )
    stored_abs = settings.data_dir / stored_rel
    _store_copy(path, stored_abs)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO sources (notebook_id, title, original_filename, "
                "stored_path, mime, sha256, size_bytes, page_count, status, error) "
                "VALUES (?, ?, ?, ?, 'application/pdf', ?, ?, ?, ?, ?)",
                (
                    notebook.id,
                    Path(path.name).stem,
                    path.name,
                    str(stored_rel),
                    sha256,
                    path.stat().st_size,
                    len(pages),
                    status,
                    error,
                ),
            )
            source_id = cur.lastrowid
            assert source_id is not None
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
                        notebook.id,
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
    except Exception:
        stored_abs.unlink(missing_ok=True)
        raise
    t = lap("sqlite", t)

    faiss_result = store.sync(notebook.id)
    lap("faiss", t)
    timings["total"] = time.perf_counter() - started

    return IngestReport(
        source_id=source_id,
        filename=path.name,
        pages=len(pages),
        pages_ocr=sum(p.method == "ocr" for p in pages),
        pages_empty=len(empty),
        chunks=len(chunks),
        faiss=faiss_result,
        timings=timings,
        warnings=warnings,
    )
