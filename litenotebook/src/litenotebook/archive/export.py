"""Write notebook exports and full backups (format version 1).

Everything streams: rows go to the ZIP one JSON line at a time and PDFs are
copied in bounded chunks while being hashed, so memory use does not grow with
notebook or file size. The archive is written next to its destination under a
temporary name and renamed into place only once complete.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import sqlite3
import sys
import time
import uuid
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from litenotebook import __version__
from litenotebook.archive.format import (
    BACKUP_SETTINGS,
    BLOB,
    CHECKSUMS,
    EMBEDDING_DTYPE,
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST,
    NOTEBOOK,
    SETTINGS,
    TABLES,
    ArchiveError,
    notebook_prefix,
)
from litenotebook.config import EMBED_DIM, EMBED_MODEL, Settings
from litenotebook.db import schema_version
from litenotebook.platform_utils.filesystem import (
    COPY_BUFFER_BYTES,
    fsync_dir,
    replace_file,
)

# Rows of one notebook, per table. Every query is scoped to the notebook.
_WHERE = {
    "sources": "WHERE notebook_id = :nb AND status IN ('ready', 'failed')",
    "pages": "WHERE source_id IN (SELECT id FROM sources WHERE notebook_id = :nb "
    "AND status IN ('ready', 'failed'))",
    "chunks": "WHERE notebook_id = :nb AND source_id IN (SELECT id FROM sources "
    "WHERE notebook_id = :nb AND status IN ('ready', 'failed'))",
    "conversations": "WHERE notebook_id = :nb",
    "messages": "WHERE conversation_id IN "
    "(SELECT id FROM conversations WHERE notebook_id = :nb)",
    "message_citations": "WHERE message_id IN (SELECT m.id FROM messages m "
    "JOIN conversations c ON c.id = m.conversation_id WHERE c.notebook_id = :nb)",
    "memory_outcomes": "WHERE notebook_id = :nb AND message_id IN (SELECT m.id FROM "
    "messages m JOIN conversations c ON c.id = m.conversation_id "
    "WHERE c.notebook_id = :nb)",
    "annotations": "WHERE notebook_id = :nb AND source_id IN (SELECT id FROM sources "
    "WHERE notebook_id = :nb AND status IN ('ready', 'failed'))",
    "annotation_rects": "WHERE annotation_id IN (SELECT a.id FROM annotations a "
    "JOIN sources s ON s.id = a.source_id WHERE a.notebook_id = :nb "
    "AND s.status IN ('ready', 'failed'))",
    "reading_state": "WHERE notebook_id = :nb AND source_id IN (SELECT id FROM "
    "sources WHERE notebook_id = :nb AND status IN ('ready', 'failed'))",
    "notes": "WHERE notebook_id = :nb",
    "note_links": "WHERE note_id IN (SELECT id FROM notes WHERE notebook_id = :nb)",
    "message_memory_refs": "WHERE message_id IN (SELECT m.id FROM messages m "
    "JOIN conversations c ON c.id = m.conversation_id WHERE c.notebook_id = :nb)",
}


@dataclass
class ExportResult:
    path: Path
    size: int
    sha256: str  # of the whole archive file
    manifest: dict[str, Any]
    seconds: float
    counts: dict[str, int] = field(default_factory=dict)


class _HashingWriter:
    """Writes one ZIP entry while hashing it; tracks every entry's checksum."""

    def __init__(self, zf: zipfile.ZipFile, stamp: tuple[int, ...]) -> None:
        self.zf = zf
        self.stamp = stamp
        self.checksums: dict[str, str] = {}

    @contextlib.contextmanager
    def entry(
        self, name: str, *, compress: bool = True, large: bool = False
    ) -> Iterator[tuple[IO[bytes], Any]]:
        info = zipfile.ZipInfo(name, date_time=self.stamp[:6])  # type: ignore[arg-type]
        info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16  # a plain file on every platform
        digest = hashlib.sha256()
        with self.zf.open(info, "w", force_zip64=large) as out:
            yield out, digest
        self.checksums[name] = digest.hexdigest()

    def write_bytes(self, name: str, data: bytes) -> None:
        with self.entry(name) as (out, digest):
            digest.update(data)
            out.write(data)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1).encode()


def _row_dict(row: sqlite3.Row, table: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in TABLES[table]:
        value = row[col.name]
        if col.kind == BLOB and value is not None:
            value = base64.b64encode(bytes(value)).decode("ascii")
        out[col.name] = value
    return out


def _busy_check(conn: sqlite3.Connection, notebook_ids: list[int]) -> None:
    marks = ",".join("?" * len(notebook_ids))
    row = conn.execute(
        f"SELECT s.original_filename, n.name FROM sources s JOIN notebooks n "
        f"ON n.id = s.notebook_id WHERE s.notebook_id IN ({marks}) "
        f"AND s.status IN ('pending', 'processing') LIMIT 1",
        notebook_ids,
    ).fetchone()
    if row is not None:
        raise ArchiveError(
            f"“{row[0]}” in {row[1]} is still being processed; wait for it to "
            "finish, then export again"
        )


def _write_notebook(
    w: _HashingWriter, conn: sqlite3.Connection, settings: Settings, nb_id: int
) -> dict[str, Any]:
    nb = conn.execute(
        "SELECT "
        + ", ".join(c.name for c in NOTEBOOK)
        + " FROM notebooks WHERE id = ?",
        (nb_id,),
    ).fetchone()
    if nb is None:
        raise ArchiveError(f"Notebook {nb_id} not found")
    prefix = notebook_prefix(nb_id)
    w.write_bytes(f"{prefix}/notebook.json", _json_bytes({k: nb[k] for k in nb.keys()}))
    counts: dict[str, int] = {}
    for table, cols in TABLES.items():
        sql = (
            f"SELECT {', '.join(c.name for c in cols)} FROM {table} "
            f"{_WHERE[table]} ORDER BY "
            + ("source_id" if table == "reading_state" else "id")
        )
        n = 0
        with w.entry(f"{prefix}/tables/{table}.jsonl") as (out, digest):
            for row in conn.execute(sql, {"nb": nb_id}):
                line = (
                    json.dumps(_row_dict(row, table), ensure_ascii=False).encode()
                    + b"\n"
                )
                digest.update(line)
                out.write(line)
                n += 1
        counts[table] = n
    counts["embedded_chunks"] = conn.execute(
        "SELECT count(*) FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE c.notebook_id = ? AND c.embedding IS NOT NULL "
        "AND s.status IN ('ready', 'failed')",
        (nb_id,),
    ).fetchone()[0]

    source_bytes = 0
    for sid, sha, filename in conn.execute(
        "SELECT id, sha256, original_filename FROM sources WHERE notebook_id = ? "
        "AND status IN ('ready', 'failed') ORDER BY id",
        (nb_id,),
    ).fetchall():
        path = settings.source_file(nb_id, sid)
        if path.is_symlink() or not path.is_file():
            raise ArchiveError(
                f"The stored PDF for “{filename}” is missing; delete that source or "
                "upload it again before exporting"
            )
        size = path.stat().st_size
        with (
            open(path, "rb") as src,
            w.entry(
                f"{prefix}/sources/{sid}.pdf", compress=False, large=size >= 1 << 30
            ) as (out, digest),
        ):
            while chunk := src.read(COPY_BUFFER_BYTES):
                digest.update(chunk)
                out.write(chunk)
        if w.checksums[f"{prefix}/sources/{sid}.pdf"] != sha:
            raise ArchiveError(
                f"The stored PDF for “{filename}” does not match its recorded "
                "checksum (the file changed on disk); not exporting a corrupt copy"
            )
        source_bytes += size
    counts["source_bytes"] = source_bytes
    return {
        "id": nb_id,
        "name": nb["name"],
        "created_at": nb["created_at"],
        "dir": prefix,
        "counts": counts,
    }


def _write_archive(
    conn: sqlite3.Connection,
    settings: Settings,
    out_path: Path,
    kind: str,
    notebook_ids: list[int],
) -> ExportResult:
    started = time.perf_counter()
    if notebook_ids:
        _busy_check(conn, notebook_ids)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex[:8]}.partial")
    now = datetime.now(UTC)
    try:
        with zipfile.ZipFile(partial, "w", allowZip64=True) as zf:
            w = _HashingWriter(zf, now.timetuple())
            notebooks = [_write_notebook(w, conn, settings, i) for i in notebook_ids]
            settings_rows: list[dict[str, str]] = []
            if kind == "backup":
                marks = ",".join("?" * len(BACKUP_SETTINGS))
                settings_rows = [
                    {"key": r[0], "value": r[1]}
                    for r in conn.execute(
                        f"SELECT key, value FROM settings WHERE key IN ({marks}) "
                        "ORDER BY key",
                        BACKUP_SETTINGS,
                    )
                ]
                w.write_bytes(SETTINGS, _json_bytes({"settings": settings_rows}))
            checksums = _json_bytes({"algorithm": "sha256", "files": w.checksums})
            totals: dict[str, int] = {}
            for nb in notebooks:
                for key, value in nb["counts"].items():
                    totals[key] = totals.get(key, 0) + value
            manifest = {
                "format": FORMAT_NAME,
                "export_format_version": FORMAT_VERSION,
                "kind": kind,
                "export_id": uuid.uuid4().hex,
                "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "app_version": __version__,
                "schema_version": schema_version(conn),
                "created_on": sys.platform,  # informational only
                "embedding": {
                    "model": EMBED_MODEL,
                    "dim": EMBED_DIM,
                    "dtype": EMBEDDING_DTYPE,
                    "normalized": True,
                },
                "notebooks": notebooks,
                "counts": {"notebooks": len(notebooks), **totals},
                "settings": [r["key"] for r in settings_rows],
                "checksums_sha256": hashlib.sha256(checksums).hexdigest(),
            }
            info = zipfile.ZipInfo(CHECKSUMS, date_time=now.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, checksums)
            info = zipfile.ZipInfo(MANIFEST, date_time=now.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, _json_bytes(manifest))
        digest = hashlib.sha256()
        with open(partial, "rb") as f:
            while chunk := f.read(COPY_BUFFER_BYTES):
                digest.update(chunk)
        replace_file(partial, out_path)
        fsync_dir(out_path.parent)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return ExportResult(
        path=out_path,
        size=out_path.stat().st_size,
        sha256=digest.hexdigest(),
        manifest=manifest,
        seconds=round(time.perf_counter() - started, 3),
        counts={"notebooks": len(notebooks), **totals},
    )


def export_notebook(
    conn: sqlite3.Connection, settings: Settings, notebook_id: int, out_path: Path
) -> ExportResult:
    """One notebook with its sources, chats, notes, annotations and memory."""
    if (
        conn.execute("SELECT 1 FROM notebooks WHERE id = ?", (notebook_id,)).fetchone()
        is None
    ):
        raise ArchiveError(f"Notebook {notebook_id} not found")
    return _write_archive(conn, settings, out_path, "notebook", [int(notebook_id)])


def create_backup(
    conn: sqlite3.Connection, settings: Settings, out_path: Path
) -> ExportResult:
    """Every notebook plus non-secret settings, in one archive."""
    ids = [r[0] for r in conn.execute("SELECT id FROM notebooks ORDER BY id")]
    return _write_archive(conn, settings, out_path, "backup", ids)


def summary(conn: sqlite3.Connection, notebook_id: int) -> dict[str, int]:
    """What an export of this notebook would contain (cheap counts)."""

    def one(sql: str) -> int:
        return int(conn.execute(sql, {"nb": notebook_id}).fetchone()[0])

    return {
        "sources": one(f"SELECT count(*) FROM sources {_WHERE['sources']}"),
        "chunks": one(f"SELECT count(*) FROM chunks {_WHERE['chunks']}"),
        "notes": one(f"SELECT count(*) FROM notes {_WHERE['notes']}"),
        "conversations": one(
            f"SELECT count(*) FROM conversations {_WHERE['conversations']}"
        ),
        "messages": one(f"SELECT count(*) FROM messages {_WHERE['messages']}"),
        "annotations": one(f"SELECT count(*) FROM annotations {_WHERE['annotations']}"),
        "source_bytes": one(
            f"SELECT coalesce(sum(size_bytes), 0) FROM sources {_WHERE['sources']}"
        ),
        "busy": one(
            "SELECT count(*) FROM sources WHERE notebook_id = :nb "
            "AND status IN ('pending', 'processing')"
        ),
    }
