"""Settings page: LLM endpoint/model/API key, retrieval top-k, discovery, test."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response

from litenotebook.services.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    RAG_TOP_K,
    TOP_K_RANGE,
    SettingsError,
    check_connection,
)
from litenotebook.web.deps import (
    app_settings,
    get_db,
    is_htmx,
    llm_transport,
    redirect,
    render,
    settings_service,
)
from litenotebook.web.navigation import notify

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]


def _context(
    conn: sqlite3.Connection, request: Request, **extra: Any
) -> dict[str, Any]:
    effective = settings_service(request, conn).effective()
    embedder = getattr(request.app.state, "embedder", None)
    return {
        "settings": effective.public(),
        "embedding": embedder.status() if embedder else {"status": "stopped"},
        "idle_timeout": request.app.state.settings.embed_worker_idle_s,
        "top_k_range": TOP_K_RANGE,
        "errors": {},
        "form": {},
        **extra,
    }


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, conn: Db, saved: int = 0) -> HTMLResponse:
    return render(request, "settings.html", _context(conn, request, saved=bool(saved)))


@router.post("/settings", response_model=None)
def save_settings(
    request: Request,
    conn: Db,
    llm_base_url: Annotated[str | None, Form()] = None,
    llm_model: Annotated[str | None, Form()] = None,
    rag_top_k: Annotated[str | None, Form()] = None,
    llm_api_key: Annotated[str | None, Form()] = None,
    clear_api_key: Annotated[str | None, Form()] = None,
) -> Response:
    service = settings_service(request, conn)
    sources = service.effective().sources

    def editable(key: str, value: str | None) -> str | None:
        return None if sources[key] == "env" else value  # env always wins

    form = {
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "rag_top_k": rag_top_k,
    }
    try:
        service.update(
            llm_base_url=editable(LLM_BASE_URL, llm_base_url),
            llm_model=editable(LLM_MODEL, llm_model),
            rag_top_k=editable(RAG_TOP_K, rag_top_k),
            llm_api_key=editable(LLM_API_KEY, llm_api_key),
            clear_api_key=bool(clear_api_key) and sources[LLM_API_KEY] != "env",
        )
    except SettingsError as e:
        context = _context(conn, request, errors=e.errors, form=form)
        template = (
            "partials/settings_form.html" if is_htmx(request) else "settings.html"
        )
        return render(request, template, context, status_code=422)
    if not is_htmx(request):
        return redirect("/settings?saved=1")
    response = render(
        request, "partials/settings_form.html", _context(conn, request, saved=True)
    )
    return notify(response, "Settings saved")


def _status(request: Request, conn: sqlite3.Connection, action: str) -> HTMLResponse:
    effective = settings_service(request, conn).effective()
    report = check_connection(effective, transport=llm_transport(request))
    status = {"report": report, "action": action, "selected": effective.llm_model}
    if is_htmx(request):
        return render(request, "partials/settings_status.html", status)
    return render(request, "settings.html", _context(conn, request, **status))


@router.post("/settings/test", response_class=HTMLResponse)
def test_connection(request: Request, conn: Db) -> HTMLResponse:
    return _status(request, conn, "test")


@router.post("/settings/models", response_class=HTMLResponse)
def refresh_models(request: Request, conn: Db) -> HTMLResponse:
    return _status(request, conn, "models")


@router.get("/settings/diagnostics", response_class=HTMLResponse)
def diagnostics_page(request: Request, conn: Db) -> HTMLResponse:
    """Read-only; never starts the embedding worker or contacts the LLM."""
    from litenotebook.diagnostics import collect, format_text

    try:
        eff = settings_service(request, conn).effective()
        llm = {
            "base_url": eff.llm_base_url,
            "model": eff.llm_model,
            "api_key_configured": bool(eff.llm_api_key),
        }
    except SettingsError:
        llm = {}
    embedder = getattr(request.app.state, "embedder", None)
    report = collect(
        app_settings(request),
        effective_llm=llm,
        worker_status=embedder.status() if embedder is not None else None,
    )
    return render(
        request,
        "diagnostics.html",
        {"nav": "settings", "report": report, "text": format_text(report)},
    )
