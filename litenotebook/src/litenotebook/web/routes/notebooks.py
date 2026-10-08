"""Notebook list, create / rename / delete, and the workspace shell.

Every action works as a plain HTML form (303 redirect) and, for HTMX
requests, returns just the fragment that changed.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response

from litenotebook.notebooks import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    NotebookError,
    NotebookNotFoundError,
    create_notebook,
    delete_notebook,
    get_summary,
    list_summaries,
    rename_notebook,
)
from litenotebook.web.deps import app_settings, get_db, is_htmx, redirect, render

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]

WORKSPACE_TABS = {
    "sources": ("Sources", "Upload and manage PDFs.", 3),
    "chat": ("Chat", "Ask questions answered from this notebook's sources.", 5),
    "notes": ("Notes", "Write notes and save answers.", 6),
    "search": ("Search", "Hybrid keyword + semantic search across sources.", 4),
}
LIMITS = {"name": MAX_NAME_LENGTH, "description": MAX_DESCRIPTION_LENGTH}


def _index(
    request: Request,
    conn: sqlite3.Connection,
    *,
    status_code: int = 200,
    **context: Any,
) -> HTMLResponse:
    return render(
        request,
        "notebooks/index.html",
        {"summaries": list_summaries(conn), "limits": LIMITS, "form": {}, **context},
        status_code=status_code,
    )


def _row(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    mode: str,
    *,
    status_code: int = 200,
    **context: Any,
) -> HTMLResponse:
    return render(
        request,
        "partials/notebook_row.html",
        {
            "s": get_summary(conn, notebook_id),
            "mode": mode,
            "limits": LIMITS,
            **context,
        },
        status_code=status_code,
    )


@router.get("/", response_class=HTMLResponse)
def index(request: Request, conn: Db) -> HTMLResponse:
    return _index(request, conn)


@router.post("/notebooks", response_model=None)
def create(
    request: Request,
    conn: Db,
    name: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
) -> Response:
    form = {"name": name, "description": description}
    try:
        notebook = create_notebook(conn, name, description)
    except NotebookError as e:
        if is_htmx(request):
            return render(
                request,
                "partials/notebook_form.html",
                {"form": form, "error": str(e), "limits": LIMITS},
                status_code=422,
            )
        return _index(request, conn, form=form, error=str(e), status_code=422)
    if not is_htmx(request):
        return redirect("/")
    return render(
        request,
        "partials/notebook_created.html",
        {
            "form": {},
            "limits": LIMITS,
            "s": get_summary(conn, notebook.id),
            "mode": "view",
            "created": notebook,
        },
    )


@router.get("/notebooks/{notebook_id}/row", response_class=HTMLResponse)
def row(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    return _row(request, conn, notebook_id, "view")


@router.get("/notebooks/{notebook_id}/rename", response_class=HTMLResponse)
def rename_form(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    if is_htmx(request):
        return _row(request, conn, notebook_id, "edit")
    get_summary(conn, notebook_id)  # 404 if missing
    return _index(request, conn, editing_id=notebook_id)


@router.post("/notebooks/{notebook_id}/rename", response_model=None)
def rename(
    request: Request, conn: Db, notebook_id: int, name: Annotated[str, Form()] = ""
) -> Response:
    try:
        rename_notebook(conn, notebook_id, name)
    except NotebookNotFoundError:
        raise
    except NotebookError as e:
        if is_htmx(request):
            return _row(
                request,
                conn,
                notebook_id,
                "edit",
                error=str(e),
                value=name,
                status_code=422,
            )
        return _index(
            request,
            conn,
            editing_id=notebook_id,
            edit_error=str(e),
            edit_value=name,
            status_code=422,
        )
    return (
        _row(request, conn, notebook_id, "view") if is_htmx(request) else redirect("/")
    )


@router.get("/notebooks/{notebook_id}/delete", response_class=HTMLResponse)
def delete_confirm(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    if is_htmx(request):
        return _row(request, conn, notebook_id, "confirm")
    get_summary(conn, notebook_id)
    return _index(request, conn, confirming_id=notebook_id)


@router.post("/notebooks/{notebook_id}/delete", response_model=None)
def delete(
    request: Request, conn: Db, notebook_id: int, confirm: Annotated[str, Form()] = ""
) -> Response:
    get_summary(conn, notebook_id)  # 404 before checking confirmation
    if confirm != str(notebook_id):
        message = "Deletion not confirmed."
        if is_htmx(request):
            return _row(
                request, conn, notebook_id, "confirm", error=message, status_code=422
            )
        return _index(
            request,
            conn,
            confirming_id=notebook_id,
            confirm_error=message,
            status_code=422,
        )
    try:
        result = delete_notebook(conn, app_settings(request), notebook_id)
    except NotebookNotFoundError:
        raise
    except NotebookError as e:
        if is_htmx(request):
            return _row(
                request, conn, notebook_id, "confirm", error=str(e), status_code=409
            )
        return _index(
            request,
            conn,
            confirming_id=notebook_id,
            confirm_error=str(e),
            status_code=409,
        )
    if not is_htmx(request):
        return redirect("/")
    return render(request, "partials/notebook_deleted.html", {"result": result})


@router.get("/notebooks/{notebook_id}", response_class=HTMLResponse)
def workspace(
    request: Request, conn: Db, notebook_id: int, tab: str = "sources"
) -> HTMLResponse:
    if tab not in WORKSPACE_TABS:
        tab = "sources"
    if tab == "sources":
        from litenotebook.web.routes.sources import panel_context

        context = panel_context(request, conn, notebook_id)
    else:
        context = {"s": get_summary(conn, notebook_id)}
    return render(
        request,
        "notebook/workspace.html",
        {**context, "tabs": WORKSPACE_TABS, "tab": tab},
    )
