"""Memory: statistics, filtered list, search, node inspector and [M#] evidence.

Everything here is SQLite-only: no MiniLM, FAISS or LLM. Every route is scoped
by the notebook id in the URL.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from litenotebook.chat import MemoryRefNotFoundError, get_memory_ref
from litenotebook.memory.graph import stats
from litenotebook.memory.search import (
    FILTERS,
    list_items,
    node_detail,
    node_id_for_key,
    search_memory,
)
from litenotebook.notebooks import get_summary
from litenotebook.web.deps import get_db, is_htmx, render

router = APIRouter(prefix="/notebooks/{notebook_id}/memory")
Db = Annotated[sqlite3.Connection, Depends(get_db)]
MAX_QUERY_CHARS = 300
UI_RESULTS = 12
UI_BUDGET = 8000


class MemoryNodeNotFoundError(LookupError):
    pass


@router.get("", response_class=HTMLResponse)
def memory_page(
    request: Request,
    conn: Db,
    notebook_id: int,
    filter: str = "all",  # noqa: A002 - query parameter name
    q: str = "",
) -> HTMLResponse:
    from litenotebook.web.routes.notebooks import WORKSPACE_TABS

    summary = get_summary(conn, notebook_id)
    which = filter if filter in FILTERS else "all"
    query = " ".join(q.split())[:MAX_QUERY_CHARS]
    search = None
    items = []
    if query:
        search = search_memory(
            conn,
            notebook_id,
            query,
            include_dead_ends=True,
            limit=UI_RESULTS,
            char_budget=UI_BUDGET,
        )
        items = search.items
    else:
        items = list_items(conn, notebook_id, which)
    context: dict[str, Any] = {
        "s": summary,
        "tabs": WORKSPACE_TABS,
        "tab": "memory",
        "stats": stats(conn, notebook_id),
        "items": items,
        "filter": which,
        "filters": FILTERS,
        "query": query,
        "search": search,
        "nb_id": notebook_id,
    }
    if is_htmx(request) and request.headers.get("hx-target") == "memory-results":
        return render(request, "partials/memory_results.html", context)
    return render(request, "notebook/workspace.html", context)


@router.get("/nodes/{node_id}", response_class=HTMLResponse)
def node_page(
    request: Request, conn: Db, notebook_id: int, node_id: int
) -> HTMLResponse:
    detail = node_detail(conn, notebook_id, node_id)
    if detail is None:
        raise MemoryNodeNotFoundError(
            f"Memory node {node_id} not found in this notebook"
        )
    context = {"node": detail, "nb_id": notebook_id}
    if is_htmx(request):
        return render(request, "partials/memory_node.html", context)
    return render(
        request,
        "notebook/memory_node.html",
        {**context, "s": get_summary(conn, notebook_id)},
    )


@router.get("/refs/{ref_id}", response_class=HTMLResponse)
def memory_ref(
    request: Request, conn: Db, notebook_id: int, ref_id: int
) -> HTMLResponse:
    ref = get_memory_ref(conn, notebook_id, ref_id)
    context = {
        "ref": ref,
        "nb_id": notebook_id,
        "live_node_id": node_id_for_key(conn, notebook_id, ref.node_key),
    }
    if is_htmx(request):
        return render(request, "partials/memory_ref.html", context)
    return render(
        request,
        "notebook/memory_ref_page.html",
        {**context, "s": get_summary(conn, notebook_id)},
    )


__all__ = ["MemoryNodeNotFoundError", "MemoryRefNotFoundError", "router"]
