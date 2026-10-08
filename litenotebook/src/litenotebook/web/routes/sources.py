"""Sources: PDF upload, status polling, details, retry and delete.

Every route is scoped by the notebook id in the URL: a source id that belongs
to another notebook is "not found". Like the notebook routes, each action works
as a plain HTML form (303 redirect) and returns only the changed fragment for
HTMX requests.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from litenotebook.config import Settings
from litenotebook.db import connect
from litenotebook.notebooks import get_notebook, get_summary
from litenotebook.services.ingestion import IngestionWorker
from litenotebook.services.uploads import ReceivedUpload, UploadError, receive_upload
from litenotebook.sources import (
    Source,
    SourceBusyError,
    SourceError,
    SourceNotFoundError,
    get_source,
    list_sources,
    reset_for_retry,
)
from litenotebook.web.deps import app_settings, get_db, is_htmx, redirect, render

router = APIRouter(prefix="/notebooks/{notebook_id}/sources")
Db = Annotated[sqlite3.Connection, Depends(get_db)]
TRUTHY = {"1", "on", "true", "yes"}


def worker(request: Request) -> IngestionWorker:
    w: IngestionWorker = request.app.state.worker
    return w


def sources_url(notebook_id: int) -> str:
    return f"/notebooks/{notebook_id}?tab=sources"


def panel_context(
    request: Request, conn: sqlite3.Connection, notebook_id: int, **extra: Any
) -> dict[str, Any]:
    settings = app_settings(request)
    return {
        "s": get_summary(conn, notebook_id),
        "sources": list_sources(conn, notebook_id),
        "max_upload_mb": settings.max_upload_mb,
        "upload": {},
        **extra,
    }


def _workspace(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    from litenotebook.web.routes.notebooks import WORKSPACE_TABS

    return render(
        request,
        "notebook/workspace.html",
        {
            **panel_context(request, conn, notebook_id, **extra),
            "tabs": WORKSPACE_TABS,
            "tab": "sources",
        },
        status_code=status_code,
    )


def _row(
    request: Request,
    conn: sqlite3.Connection,
    source: Source,
    mode: str = "view",
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    return render(
        request,
        "partials/source_row.html",
        {
            "src": source,
            "mode": mode,
            "s": get_summary(conn, source.notebook_id),
            "oob_counts": True,
            **extra,
        },
        status_code=status_code,
    )


# --- upload ----------------------------------------------------------------------


def _register(settings: Settings, notebook_id: int, upload: ReceivedUpload) -> int:
    """Validate + store the streamed file and record a pending source (thread)."""
    from litenotebook.ingest.pipeline import (
        DuplicateSourceError,
        IngestError,
        UnsupportedFileError,
        register_source,
    )

    conn = connect(settings.db_path)
    try:
        notebook = get_notebook(conn, notebook_id)
        source = register_source(
            conn,
            settings,
            notebook,
            upload.path,
            filename=upload.filename,
            ocr=upload.fields.get("ocr", "").strip().lower() in TRUTHY,
            move=True,
        )
        return source.id
    except UnsupportedFileError as e:
        raise UploadError(str(e), 415) from e
    except DuplicateSourceError as e:
        raise UploadError(str(e), 409) from e
    except IngestError as e:
        raise UploadError(str(e), 422) from e
    finally:
        upload.discard()  # no-op once moved into place
        conn.close()


@router.post("", response_model=None)
async def upload(request: Request, notebook_id: int) -> Response:
    settings = app_settings(request)

    def check_notebook() -> None:
        conn = connect(settings.db_path)
        try:
            get_notebook(conn, notebook_id)  # 404 before reading the body
        finally:
            conn.close()

    await run_in_threadpool(check_notebook)
    try:
        received = await receive_upload(request, settings)
        source_id = await run_in_threadpool(_register, settings, notebook_id, received)
    except UploadError as e:
        return await run_in_threadpool(
            _upload_error, request, settings, notebook_id, str(e), e.status_code
        )
    worker(request).enqueue(source_id)
    if not is_htmx(request):
        return redirect(sources_url(notebook_id))

    def created() -> HTMLResponse:
        conn = connect(settings.db_path)
        try:
            return render(
                request,
                "partials/source_created.html",
                {
                    **panel_context(request, conn, notebook_id),
                    "src": get_source(conn, notebook_id, source_id),
                    "mode": "view",
                },
            )
        finally:
            conn.close()

    return await run_in_threadpool(created)


def _upload_error(
    request: Request,
    settings: Settings,
    notebook_id: int,
    message: str,
    status_code: int,
) -> HTMLResponse:
    conn = connect(settings.db_path)
    try:
        if is_htmx(request):
            return render(
                request,
                "partials/source_upload.html",
                panel_context(request, conn, notebook_id, upload_error=message),
                status_code=status_code,
            )
        return _workspace(
            request, conn, notebook_id, upload_error=message, status_code=status_code
        )
    finally:
        conn.close()


# --- per-source views ------------------------------------------------------------


@router.get("/{source_id}", response_class=HTMLResponse)
def detail(
    request: Request, conn: Db, notebook_id: int, source_id: int
) -> HTMLResponse:
    source = get_source(conn, notebook_id, source_id)
    return render(
        request,
        "notebook/source.html",
        {
            "s": get_summary(conn, notebook_id),
            "src": source,
            "confirming": request.query_params.get("confirm") == "delete",
        },
    )


@router.get("/{source_id}/status", response_class=HTMLResponse)
def status(
    request: Request, conn: Db, notebook_id: int, source_id: int
) -> HTMLResponse:
    """Row fragment for HTMX polling; polling stops once the source settles."""
    return _row(request, conn, get_source(conn, notebook_id, source_id))


@router.post("/{source_id}/retry", response_model=None)
def retry(request: Request, conn: Db, notebook_id: int, source_id: int) -> Response:
    try:
        source = reset_for_retry(conn, app_settings(request), notebook_id, source_id)
    except SourceNotFoundError:
        raise
    except SourceBusyError as e:
        return _action_error(request, conn, notebook_id, source_id, str(e), 409)
    except SourceError as e:
        return _action_error(request, conn, notebook_id, source_id, str(e), 422)
    worker(request).enqueue(source.id)
    if not is_htmx(request):
        return redirect(sources_url(notebook_id))
    return _row(request, conn, source)


@router.get("/{source_id}/delete", response_class=HTMLResponse)
def delete_confirm(
    request: Request, conn: Db, notebook_id: int, source_id: int
) -> Response:
    source = get_source(conn, notebook_id, source_id)
    if not is_htmx(request):
        return redirect(f"/notebooks/{notebook_id}/sources/{source_id}?confirm=delete")
    return _row(request, conn, source, "confirm")


@router.post("/{source_id}/delete", response_model=None)
def delete(
    request: Request,
    conn: Db,
    notebook_id: int,
    source_id: int,
    confirm: Annotated[str, Form()] = "",
) -> Response:
    get_source(conn, notebook_id, source_id)  # 404 before anything else
    if confirm != str(source_id):
        return _action_error(
            request,
            conn,
            notebook_id,
            source_id,
            "Deletion not confirmed.",
            422,
            mode="confirm",
        )
    from litenotebook.index.faiss_store import FaissStore
    from litenotebook.ingest.pipeline import delete_source

    settings = app_settings(request)
    try:
        deleted = delete_source(
            conn, settings, notebook_id, source_id, FaissStore(conn, settings)
        )
    except SourceBusyError as e:
        return _action_error(
            request, conn, notebook_id, source_id, str(e), 409, mode="confirm"
        )
    if not is_htmx(request):
        return redirect(sources_url(notebook_id))
    return render(
        request,
        "partials/source_deleted.html",
        {"src": deleted, "s": get_summary(conn, notebook_id), "oob_counts": True},
    )


def _action_error(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    source_id: int,
    message: str,
    status_code: int,
    *,
    mode: str = "view",
) -> HTMLResponse:
    source = get_source(conn, notebook_id, source_id)
    if is_htmx(request):
        return _row(request, conn, source, mode, error=message, status_code=status_code)
    return render(
        request,
        "notebook/source.html",
        {
            "s": get_summary(conn, notebook_id),
            "src": source,
            "confirming": mode == "confirm",
            "error": message,
        },
        status_code=status_code,
    )
