"""Memory: statistics, filtered list, search, node inspector and [M#] evidence.

Everything here is SQLite-only: no MiniLM, FAISS or LLM. Every route is scoped
by the notebook id in the URL.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

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
from litenotebook.web.navigation import last_read

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
        "last_read": last_read(conn, notebook_id),
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


@router.get("/graph", response_class=HTMLResponse)
def graph_page(
    request: Request,
    conn: Db,
    notebook_id: int,
    node: str | None = None,
    depth: int = 1,
    mode: str = "",
) -> HTMLResponse:
    """The knowledge graph: Global (whole notebook, bounded) or Local (one
    node's neighbourhood). ``node`` may be a view id ("m:12", "s:3") or a
    memory node number (links from the memory inspector)."""
    from litenotebook.memory import visual
    from litenotebook.web.routes.notebooks import WORKSPACE_TABS

    summary = get_summary(conn, notebook_id)
    center = None
    if node:
        center = f"m:{node}" if node.isdigit() else node
        if visual.node_kind(conn, notebook_id, center) is None:
            raise MemoryNodeNotFoundError("That graph node is not in this notebook")
    which = mode if mode in ("global", "local") else ("local" if center else "global")
    return render(
        request,
        "notebook/workspace.html",
        {
            "s": summary,
            "tabs": WORKSPACE_TABS,
            "last_read": last_read(conn, notebook_id),
            "tab": "memory",
            "memory_view": "graph",
            "graph_mode": which,
            "center": center or "",
            "depth": max(1, min(visual.MAX_DEPTH, depth)),
            "type_labels": visual.TYPE_LABELS,
            "default_types": visual.DEFAULT_TYPES,
            "optional_types": visual.OPTIONAL_TYPES,
            "stats": stats(conn, notebook_id),
            "nb_id": notebook_id,
        },
    )


def _flag(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


@router.get("/graph/view.json", response_model=None)
def graph_view(
    conn: Db,
    notebook_id: int,
    types: str | None = None,
    min_degree: int = 0,
    hide_orphans: str = "0",
    concept_min: str = "auto",
    max_nodes: int = 0,
) -> JSONResponse:
    """Global graph: structured nodes/edges only (labels, ids, flags)."""
    from litenotebook.memory import visual

    get_summary(conn, notebook_id)
    return JSONResponse(
        visual.global_graph(
            conn,
            notebook_id,
            visual.parse_types(types),
            min_degree=max(0, min(50, min_degree)),
            hide_orphans=_flag(hide_orphans),
            concept_min=int(concept_min) if concept_min.isdigit() else None,
            max_nodes=max_nodes or visual.GLOBAL_NODES,
        )
    )


@router.get("/graph/local.json", response_model=None)
def graph_local(
    conn: Db, notebook_id: int, node: str, depth: int = 1, types: str | None = None
) -> JSONResponse:
    from litenotebook.memory import visual

    get_summary(conn, notebook_id)
    graph = visual.local_graph(
        conn, notebook_id, node[:40], depth, visual.parse_types(types)
    )
    if graph is None:
        return JSONResponse({"error": "That graph node is not in this notebook"}, 404)
    return JSONResponse(graph)


@router.get("/graph/node.json", response_model=None)
def graph_node(conn: Db, notebook_id: int, id: str) -> JSONResponse:  # noqa: A002
    from litenotebook.memory import visual

    get_summary(conn, notebook_id)
    detail = visual.node_detail(conn, notebook_id, id[:40])
    if detail is None:
        return JSONResponse({"error": "That graph node is not in this notebook"}, 404)
    return JSONResponse(detail)


@router.get("/graph/find.json", response_model=None)
def graph_find(
    conn: Db, notebook_id: int, q: str = "", types: str | None = None
) -> JSONResponse:
    """Label search for the graph (substring; no AI)."""
    from litenotebook.memory import visual

    get_summary(conn, notebook_id)
    wanted = visual.parse_types(types, visual.ALL_TYPES)
    return JSONResponse({"results": visual.search(conn, notebook_id, q, wanted)})


@router.get("/graph.json", response_model=None)
def graph_json(
    conn: Db,
    notebook_id: int,
    node: int,
    depth: int = 1,
    types: str = "",
) -> JSONResponse:
    from litenotebook.memory.explorer import subgraph

    get_summary(conn, notebook_id)
    wanted = {t for t in types.split(",") if t} or None
    graph = subgraph(conn, notebook_id, node, depth, wanted)
    if graph is None:
        return JSONResponse({"error": "Memory node not found in this notebook"}, 404)
    return JSONResponse(graph)


@router.get("/graph/search", response_model=None)
def graph_search(conn: Db, notebook_id: int, q: str = "") -> JSONResponse:
    from litenotebook.memory.explorer import search_nodes

    get_summary(conn, notebook_id)
    return JSONResponse({"results": search_nodes(conn, notebook_id, q)})


@router.get("/edges/{edge_id}", response_class=HTMLResponse)
def edge_page(
    request: Request, conn: Db, notebook_id: int, edge_id: int
) -> HTMLResponse:
    from litenotebook.memory.explorer import edge_detail

    edge = edge_detail(conn, notebook_id, edge_id)
    if edge is None:
        raise MemoryNodeNotFoundError(
            f"Memory link {edge_id} not found in this notebook"
        )
    return render(
        request, "partials/memory_edge.html", {"edge": edge, "nb_id": notebook_id}
    )


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
