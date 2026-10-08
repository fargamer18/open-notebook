"""Application shell: command-palette data and the global Search page.

Both read SQLite only. The palette never calls Qwen or MiniLM, and global
search is keyword-only (FTS5), so neither can wake the embedding worker.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from litenotebook.db import connect
from litenotebook.notebooks import NotebookNotFoundError, get_notebook, list_notebooks
from litenotebook.unified_search import MAX_QUERY_CHARS, clean_query, unified_search
from litenotebook.web.deps import app_settings, get_db, render
from litenotebook.web.navigation import last_read

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]
MAX_PALETTE_NOTEBOOKS = 200
MAX_PALETTE_SOURCES = 50
GLOBAL_HITS_PER_NOTEBOOK = 3


def _cmd(
    group: str, label: str, url: str, keywords: str = "", **extra: Any
) -> dict[str, Any]:
    return {"group": group, "label": label, "url": url, "keywords": keywords, **extra}


def palette_commands(
    conn: sqlite3.Connection, notebook_id: int | None, source_id: int | None
) -> list[dict[str, Any]]:
    """Commands in display order: notebook actions first when inside one."""
    commands: list[dict[str, Any]] = []
    if notebook_id is not None:
        try:
            nb = get_notebook(conn, notebook_id)
        except NotebookNotFoundError:
            nb = None
        if nb is not None:
            base = f"/notebooks/{nb.id}"
            group = nb.name
            if source_id is not None:
                row = conn.execute(
                    "SELECT original_filename FROM sources WHERE id = ? "
                    "AND notebook_id = ? AND status = 'ready'",
                    (source_id, nb.id),
                ).fetchone()
                if row:
                    commands.append(
                        _cmd(
                            group,
                            f"Open current PDF in reader: {row[0]}",
                            f"{base}/sources/{source_id}/read",
                            "reader pdf",
                        )
                    )
            recent = last_read(conn, nb.id)
            if recent and recent["id"] != source_id:
                commands.append(
                    _cmd(
                        group,
                        f"Continue reading: {recent['filename']}",
                        f"{base}/sources/{recent['id']}/read",
                        "reader pdf resume",
                    )
                )
            commands += [
                _cmd(
                    group, "New chat", f"{base}/chat/new", "ask question conversation"
                ),
                _cmd(group, "New note", f"{base}/notes?new=1", "write create"),
                _cmd(group, "Search this notebook", f"{base}/search", "find"),
                _cmd(group, "Sources", f"{base}?tab=sources", "pdf upload files"),
                _cmd(group, "Chat", f"{base}/chat", "conversations ask"),
                _cmd(group, "Notes", f"{base}/notes", "write"),
                _cmd(group, "Memory", f"{base}/memory", "remembered history"),
                _cmd(group, "Memory graph", f"{base}/memory/graph", "concepts map"),
                _cmd(group, "Export this notebook", f"{base}/export", "backup zip"),
            ]
            for sid, name in conn.execute(
                "SELECT id, original_filename FROM sources WHERE notebook_id = ? "
                "AND status = 'ready' ORDER BY created_at DESC, id DESC LIMIT ?",
                (nb.id, MAX_PALETTE_SOURCES),
            ):
                if sid != source_id:
                    commands.append(
                        _cmd(
                            group, f"Read: {name}", f"{base}/sources/{sid}/read", "pdf"
                        )
                    )
    commands += [
        _cmd("LiteNotebook", "Notebooks", "/", "home list"),
        _cmd("LiteNotebook", "New notebook", "/?new=1", "create"),
        _cmd("LiteNotebook", "Search all notebooks", "/search", "find"),
        _cmd("LiteNotebook", "Settings", "/settings", "llm model status"),
        _cmd("LiteNotebook", "Backup & Restore", "/backup", "export import verify zip"),
        _cmd("LiteNotebook", "Keyboard shortcuts", "", "help keys", action="shortcuts"),
    ]
    for nb in list_notebooks(conn)[:MAX_PALETTE_NOTEBOOKS]:
        if nb.id != notebook_id:
            commands.append(
                _cmd(
                    "Notebooks",
                    f"Open notebook: {nb.name}",
                    f"/notebooks/{nb.id}",
                    "switch",
                )
            )
    return commands


@router.get("/palette.json", response_model=None)
def palette(conn: Db, nb: int | None = None, source: int | None = None) -> JSONResponse:
    return JSONResponse(
        {"commands": palette_commands(conn, nb, source)},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/search", response_class=HTMLResponse)
async def global_search(request: Request, q: str = "") -> HTMLResponse:
    """Keyword search across every notebook, grouped by notebook."""
    settings = app_settings(request)
    query = clean_query(q)

    def run() -> HTMLResponse:
        conn = connect(settings.db_path)
        try:
            started = time.perf_counter()
            groups = []
            notebooks = list_notebooks(conn)
            if query:
                for nb in notebooks:
                    results = unified_search(conn, settings, nb.id, query)
                    hits = [h for sec in results.sections.values() for h in sec]
                    if hits:
                        groups.append(
                            {
                                "notebook": nb,
                                "hits": hits[:GLOBAL_HITS_PER_NOTEBOOK],
                                "total": results.total,
                            }
                        )
            return render(
                request,
                "search.html",
                {
                    "nav": "search",
                    "query": query,
                    "groups": groups,
                    "notebook_count": len(notebooks),
                    "max_query": MAX_QUERY_CHARS,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        finally:
            conn.close()

    return await run_in_threadpool(run)
