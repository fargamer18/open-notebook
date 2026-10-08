"""Unified Search workspace (sources, notes, chats, memory).

Searching happens only on an explicit submit (never per keystroke). Lexical
search never starts the embedding worker; hybrid source search may, and the
page says so before you run it.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from litenotebook.db import connect
from litenotebook.notebooks import get_summary
from litenotebook.unified_search import ENGINES, MAX_QUERY_CHARS, MODES, unified_search
from litenotebook.web.deps import app_settings, get_db, is_htmx, render
from litenotebook.web.navigation import last_read

router = APIRouter(prefix="/notebooks/{notebook_id}/search")
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.get("", response_class=HTMLResponse)
async def search_page(
    request: Request,
    notebook_id: int,
    q: str = "",
    mode: str = "all",
    engine: str = "lexical",
) -> HTMLResponse:
    from litenotebook.web.routes.notebooks import WORKSPACE_TABS

    settings = app_settings(request)
    embedder = request.app.state.embedder
    mode = mode if mode in MODES else "all"
    engine = engine if engine in ENGINES else "lexical"
    warm_before = embedder.loaded

    def run() -> HTMLResponse:
        conn = connect(settings.db_path)
        try:
            summary = get_summary(conn, notebook_id)
            started = time.perf_counter()
            results = unified_search(
                conn,
                settings,
                notebook_id,
                q,
                mode=mode,
                engine=engine,
                embedder=embedder if engine == "hybrid" else None,
            )
            context = {
                "s": summary,
                "tabs": WORKSPACE_TABS,
                "last_read": last_read(conn, notebook_id),
                "tab": "search",
                "results": results,
                "query": results.query,
                "mode": mode,
                "engine": engine,
                "modes": MODES,
                "embedder_ready": embedder.loaded,
                "cold_start": engine == "hybrid"
                and bool(results.query)
                and not warm_before
                and embedder.loaded,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                "max_query": MAX_QUERY_CHARS,
                "nb_id": notebook_id,
            }
            if (
                is_htmx(request)
                and request.headers.get("hx-target") == "search-results"
            ):
                return render(request, "partials/search_results.html", context)
            return render(request, "notebook/workspace.html", context)
        finally:
            conn.close()

    return await run_in_threadpool(run)
