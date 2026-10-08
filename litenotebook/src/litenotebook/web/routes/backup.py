"""Backup & Restore: notebook export/import, full backups, verify, restore.

Uploaded archives stream to ``<data>/tmp/archives`` under the archive size
limit (never into RAM) and are deleted afterwards. Downloads are written to
the same place and removed once sent. No page shows a filesystem path, and
nothing here loads MiniLM or calls the LLM.
"""

from __future__ import annotations

import contextlib
import re
import secrets
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from litenotebook.archive.format import ArchiveError, Limits
from litenotebook.config import Settings
from litenotebook.db import connect, schema_version
from litenotebook.notebooks import get_summary, list_summaries
from litenotebook.platform_utils.paths import download_name
from litenotebook.services.uploads import UploadError, receive_upload
from litenotebook.web.deps import app_settings, get_db, redirect, render
from litenotebook.web.navigation import last_read

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]
CONFIRM_WORD = "RESTORE"
_TOKEN = re.compile(r"^[0-9a-f]{32}$")


def _page(
    request: Request,
    conn: sqlite3.Connection,
    *,
    status_code: int = 200,
    **context: Any,
) -> HTMLResponse:
    from litenotebook.archive.restore import pending_restore, safety_backups

    settings = app_settings(request)
    return render(
        request,
        "backup.html",
        {
            "nav": "settings",
            "summaries": list_summaries(conn),
            "pending": pending_restore(settings.data_dir),
            "safety_count": len(safety_backups(settings.data_dir)),
            "max_archive_mb": settings.archive_max_mb,
            **context,
        },
        status_code=status_code,
    )


def _download(path: Path, filename: str) -> FileResponse:
    return FileResponse(
        path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(path.unlink, missing_ok=True),
        headers={"Cache-Control": "no-store"},
    )


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M")


def _tmp_zip(settings: Settings, prefix: str) -> Path:
    settings.archive_tmp_dir.mkdir(parents=True, exist_ok=True)
    return settings.archive_tmp_dir / f"{prefix}-{secrets.token_hex(16)}.zip"


async def _receive(request: Request) -> tuple[Path, dict[str, str]]:
    settings = app_settings(request)
    upload = await receive_upload(
        request,
        settings,
        max_bytes=settings.archive_max_mb * 1024 * 1024,
        kind="archive",
        tmp_dir=settings.archive_tmp_dir,
    )
    return upload.path, upload.fields


# --- pages -------------------------------------------------------------------------


@router.get("/backup", response_class=HTMLResponse)
def backup_page(request: Request, conn: Db) -> HTMLResponse:
    return _page(request, conn)


@router.get("/notebooks/{notebook_id}/export", response_class=HTMLResponse)
def export_page(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    from litenotebook.archive.export import summary
    from litenotebook.web.routes.notebooks import WORKSPACE_TABS

    s = get_summary(conn, notebook_id)
    return render(
        request,
        "notebook/export.html",
        {
            "nav": "notebooks",
            "s": s,
            "tabs": WORKSPACE_TABS,
            "tab": "export",
            "last_read": last_read(conn, notebook_id),
            "summary": summary(conn, notebook_id),
        },
    )


# --- downloads ---------------------------------------------------------------------


@router.post("/notebooks/{notebook_id}/export", response_model=None)
def export_download(request: Request, conn: Db, notebook_id: int) -> Response:
    from litenotebook.archive.export import export_notebook

    settings = app_settings(request)
    s = get_summary(conn, notebook_id)
    path = _tmp_zip(settings, "export")
    try:
        export_notebook(conn, settings, notebook_id, path)
    except ArchiveError as e:
        path.unlink(missing_ok=True)
        return _page(request, conn, error=str(e), status_code=409)
    name = f"litenotebook-{download_name(s.notebook.name)}-{_stamp()}.zip"
    return _download(path, name)


@router.post("/backup/create", response_model=None)
def backup_download(request: Request, conn: Db) -> Response:
    from litenotebook.archive.export import create_backup

    settings = app_settings(request)
    path = _tmp_zip(settings, "backup")
    try:
        create_backup(conn, settings, path)
    except ArchiveError as e:
        path.unlink(missing_ok=True)
        return _page(request, conn, error=str(e), status_code=409)
    return _download(path, f"litenotebook-backup-{_stamp()}.zip")


# --- uploads -----------------------------------------------------------------------


async def _with_upload(request: Request, work: Any, *, keep: bool = False) -> Response:
    """Receive an archive, run ``work(conn, settings, path, fields)`` in a
    worker thread, always delete the upload unless ``keep``."""
    settings = app_settings(request)
    try:
        path, fields = await _receive(request)
    except UploadError as e:
        message, status = str(e), e.status_code

        def failed() -> HTMLResponse:
            conn = connect(settings.db_path)
            try:
                return _page(request, conn, error=message, status_code=status)
            finally:
                conn.close()

        return await run_in_threadpool(failed)

    def run() -> Response:
        conn = connect(settings.db_path)
        keep_file = keep
        try:
            response: Response = work(conn, settings, path, fields)
            return response
        except ArchiveError as e:
            keep_file = False
            return _page(request, conn, error=str(e), status_code=422)
        finally:
            conn.close()
            if not keep_file:
                path.unlink(missing_ok=True)

    return await run_in_threadpool(run)


@router.post("/backup/import", response_model=None)
async def import_upload(request: Request) -> Response:
    from litenotebook.archive.importer import import_archive

    def work(
        conn: sqlite3.Connection, settings: Settings, path: Path, fields: dict[str, str]
    ) -> Response:
        name = (fields.get("name") or "").strip() or None
        stages: list[str] = []
        report, results = import_archive(
            conn, settings, path, name=name, progress=stages.append
        )
        return render(
            request,
            "backup_result.html",
            {
                "nav": "settings",
                "mode": "import",
                "verify": report,
                "results": results,
                "stages": stages,
            },
        )

    return await _with_upload(request, work)


@router.post("/backup/verify", response_model=None)
async def verify_upload(request: Request) -> Response:
    from litenotebook.archive.reader import verify_archive

    def work(
        conn: sqlite3.Connection, settings: Settings, path: Path, _f: dict[str, str]
    ) -> Response:
        report = verify_archive(
            path, Limits.from_settings(settings), current_schema=schema_version(conn)
        )
        return render(
            request,
            "backup_result.html",
            {"nav": "settings", "mode": "verify", "verify": report},
        )

    return await _with_upload(request, work)


@router.post("/backup/restore/preview", response_model=None)
async def restore_preview(request: Request) -> Response:
    from litenotebook.archive.reader import verify_archive
    from litenotebook.archive.restore import preview

    def work(
        conn: sqlite3.Connection, settings: Settings, path: Path, _f: dict[str, str]
    ) -> Response:
        report = verify_archive(
            path, Limits.from_settings(settings), current_schema=schema_version(conn)
        )
        if report.kind != "backup":
            raise ArchiveError(
                "This is a single-notebook export; use Import notebook instead"
            )
        token = secrets.token_hex(16)
        path.rename(settings.archive_tmp_dir / f"restore-{token}.zip")
        return render(
            request,
            "backup_result.html",
            {
                "nav": "settings",
                "mode": "restore-preview",
                "verify": report,
                "preview": preview(report),
                "token": token,
                "confirm_word": CONFIRM_WORD,
            },
        )

    return await _with_upload(request, work)


@router.post("/backup/restore/confirm", response_model=None)
def restore_confirm(
    request: Request,
    conn: Db,
    token: Annotated[str, Form()] = "",
    confirm: Annotated[str, Form()] = "",
    understand: Annotated[str, Form()] = "",
) -> Response:
    from litenotebook.archive.restore import stage_restore

    settings = app_settings(request)
    if not _TOKEN.match(token):
        return _page(
            request,
            conn,
            error="That restore request expired; upload the backup again",
            status_code=422,
        )
    path = settings.archive_tmp_dir / f"restore-{token}.zip"
    if not path.is_file():
        return _page(
            request,
            conn,
            error="That restore request expired; upload the backup again",
            status_code=422,
        )
    if confirm.strip() != CONFIRM_WORD or understand != "1":
        path.unlink(missing_ok=True)
        return _page(
            request,
            conn,
            error=f"Restore not confirmed: type {CONFIRM_WORD} "
            "and tick the box. Upload the backup again to retry.",
            status_code=422,
        )
    try:
        staged = stage_restore(settings, path)
    except ArchiveError as e:
        return _page(request, conn, error=str(e), status_code=422)
    finally:
        path.unlink(missing_ok=True)
    return render(
        request,
        "backup_result.html",
        {"nav": "settings", "mode": "restore-staged", "staged": staged.marker},
    )


@router.post("/backup/restore/cancel", response_model=None)
def restore_cancel(request: Request) -> Response:
    from litenotebook.archive.restore import cancel_restore

    with contextlib.suppress(OSError):
        cancel_restore(app_settings(request).data_dir)
    return redirect("/backup", flash="restore-cancelled")
