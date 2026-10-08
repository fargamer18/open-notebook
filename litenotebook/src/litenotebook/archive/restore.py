"""Full restore from a backup: staged, verified, then switched in.

The live data directory is never modified in place:

1. **Verify** the whole backup (checksums, rows, PDFs) — nothing written.
2. **Stage**: build a complete new data directory next to the live one,
   ``<data>.restore-staging/``, by importing every notebook (names kept,
   new ids) and the non-secret settings into a fresh database, rebuilding
   FAISS and the memory graph there.
3. **Check** the staged database (SQLite integrity, foreign keys, FTS) and
   every staged notebook (FAISS n/n, graph, files). Only then write the
   ``RESTORE-READY.json`` marker.
4. **Switch** (``apply_pending_restore``), only while no LiteNotebook server
   holds the data directory's instance lock:
   rename ``<data>`` → ``<data>.pre-restore-<UTC time>`` (the safety backup,
   kept until you delete it), then rename the staging directory → ``<data>``.
   If the second rename fails the first is undone.

Why rename-on-start instead of swapping a running server's files: Windows
cannot rename a directory while files in it are open, and a POSIX rename
under a running server would leave it writing to the old files. So a
restore requested from the web UI is staged, then applied the next time
LiteNotebook starts; the CLI applies it immediately when no server is
running. An unmarked (incomplete) staging directory is discarded.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from litenotebook.archive.format import ArchiveError, Limits
from litenotebook.archive.importer import Progress, import_notebook
from litenotebook.archive.integrity import database_report
from litenotebook.archive.reader import ArchiveReader, VerifyReport
from litenotebook.config import Settings
from litenotebook.db import available_migrations, open_db
from litenotebook.platform_utils.filesystem import in_use, remove_tree, rename_dir

MARKER = "RESTORE-READY.json"
STAGING_SUFFIX = ".restore-staging"
SAFETY_SUFFIX = ".pre-restore-"


class RestoreBusyError(ArchiveError):
    pass


@dataclass
class StagedRestore:
    staging: Path
    marker: dict[str, Any]
    notebooks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AppliedRestore:
    data_dir: Path
    safety_dir: Path | None
    marker: dict[str, Any]


def staging_dir(data_dir: Path) -> Path:
    return data_dir.with_name(data_dir.name + STAGING_SUFFIX)


def pending_restore(data_dir: Path) -> dict[str, Any] | None:
    marker = staging_dir(data_dir) / MARKER
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def cancel_restore(data_dir: Path) -> bool:
    staging = staging_dir(data_dir)
    if not staging.exists():
        return False
    remove_tree(staging)
    return True


def preview(report: VerifyReport) -> dict[str, Any]:
    m = report.manifest
    return {
        "kind": m.get("kind"),
        "format_version": m.get("export_format_version"),
        "schema_version": m.get("schema_version"),
        "app_version": m.get("app_version"),
        "created_at": m.get("created_at"),
        "notebooks": [
            {
                "name": nb.meta["name"],
                "sources": nb.counts.get("sources", 0),
                "chunks": nb.counts.get("chunks", 0),
            }
            for nb in report.notebooks
        ],
        "sources": sum(nb.counts.get("sources", 0) for nb in report.notebooks),
        "size": report.size,
        "sha256": report.sha256,
        "warnings": report.warnings,
    }


def stage_restore(
    settings: Settings, archive: Path, *, progress: Progress | None = None
) -> StagedRestore:
    say = progress or (lambda _stage: None)
    data_dir = settings.data_dir
    staging = staging_dir(data_dir)
    if pending_restore(data_dir) is not None:
        raise ArchiveError(
            "A restore is already staged; it is applied when LiteNotebook next "
            "starts (or cancel it first)"
        )
    if staging.exists():  # an interrupted earlier attempt
        remove_tree(staging)
    staged = replace(settings, data_dir=staging)
    say("Validating…")
    reader = ArchiveReader(archive, Limits.from_settings(settings))
    try:
        reader.__enter__()
        report = reader.validate(current_schema=len(available_migrations()))
        if report.kind != "backup":
            raise ArchiveError(
                "This is a single-notebook export; use Import notebook, not Restore"
            )
        staging.mkdir(parents=True)
        conn = open_db(staged.db_path)
        try:
            results = []
            for info in report.notebooks:
                result = import_notebook(
                    conn, staged, reader, info, keep_name=True, progress=say
                )
                results.append(
                    {
                        "name": result.name,
                        **result.report.counts,
                        "faiss": result.report.faiss,
                    }
                )
            with conn:
                conn.executemany(
                    "INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    [
                        (r["key"], r["value"])
                        for r in report.settings
                        if r["key"] in ("llm.base_url", "llm.model", "rag.top_k")
                    ],
                )
            _carry_local_secrets(settings, conn)
            say("Checking integrity…")
            db = database_report(conn)
            if not db.ok:
                raise ArchiveError(
                    f"The restored database failed its integrity check "
                    f"({db.integrity}; {db.foreign_keys} foreign-key problems; "
                    f"{'; '.join(db.fts_problems)})"
                )
        finally:
            conn.close()
        with contextlib.suppress(OSError):
            remove_tree(staging / "tmp")
        marker = {
            "archive_sha256": report.sha256,
            "archive_created_at": report.manifest.get("created_at"),
            "staged_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "notebooks": results,
        }
        (staging / MARKER).write_text(json.dumps(marker, indent=1), encoding="utf-8")
    except BaseException:
        reader.close()
        with contextlib.suppress(OSError):
            remove_tree(staging)
        raise
    reader.close()
    return StagedRestore(staging, marker, results)


def _carry_local_secrets(settings: Settings, staged: sqlite3.Connection) -> None:
    """Backups never contain secrets; keep this machine's API key (if any) so a
    restore does not silently drop it. It never leaves this computer."""
    if not settings.db_path.exists():
        return
    current = sqlite3.connect(settings.db_path)  # no URI: paths may contain '#' or '?'
    try:
        current.execute("PRAGMA query_only = ON")
        row = current.execute(
            "SELECT value FROM settings WHERE key = 'llm.api_key'"
        ).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        current.close()
    if row is not None:
        with staged:
            staged.execute(
                "INSERT INTO settings (key, value) VALUES ('llm.api_key', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (row[0],),
            )


def apply_pending_restore(data_dir: Path) -> AppliedRestore | None:
    """Switch a staged restore in, if one is ready and nothing uses the data.
    Raises ``RestoreBusyError`` while a server holds the data directory."""
    staging = staging_dir(data_dir)
    if not staging.exists():
        return None
    marker = pending_restore(data_dir)
    if marker is None:  # incomplete staging from an interrupted run
        remove_tree(staging)
        return None
    if data_dir.exists() and in_use(data_dir):
        raise RestoreBusyError(
            "LiteNotebook is running with this data directory; the restore will be "
            "applied the next time it starts"
        )
    safety = None
    if data_dir.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        safety = data_dir.with_name(f"{data_dir.name}{SAFETY_SUFFIX}{stamp}")
        rename_dir(data_dir, safety)
    try:
        rename_dir(staging, data_dir)
    except BaseException:
        if safety is not None:
            rename_dir(safety, data_dir)
        raise
    with contextlib.suppress(OSError):
        (data_dir / MARKER).unlink()
    return AppliedRestore(data_dir, safety, marker)


def safety_backups(data_dir: Path) -> list[Path]:
    return sorted(data_dir.parent.glob(f"{data_dir.name}{SAFETY_SUFFIX}*"))


def database_summary(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "notebooks": conn.execute("SELECT count(*) FROM notebooks").fetchone()[0],
        "sources": conn.execute("SELECT count(*) FROM sources").fetchone()[0],
    }
