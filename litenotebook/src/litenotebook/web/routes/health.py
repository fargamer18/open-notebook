"""Local health status. An unreachable LLM degrades, but never fails, the app."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from litenotebook.db import connect, schema_version
from litenotebook.services.settings import check_connection
from litenotebook.web.deps import app_settings, llm_transport, render, settings_service

router = APIRouter()
HEALTH_LLM_TIMEOUT_S = 2.0


def collect_health(request: Request) -> dict[str, Any]:
    """Never raises: each subsystem reports its own status.

    The connection is opened here rather than via the ``get_db`` dependency so
    that a broken database is reported instead of turning into a 500.
    """
    settings = app_settings(request)
    health: dict[str, Any] = {}
    conn: sqlite3.Connection | None = None
    try:
        conn = connect(settings.db_path)
        conn.execute("SELECT 1").fetchone()
        health["database"] = {"status": "ok", "schema_version": schema_version(conn)}
    except (sqlite3.Error, OSError) as e:
        health["database"] = {"status": "error", "error": str(e)}
        if conn is not None:
            conn.close()
        conn = None

    data_dir = settings.data_dir
    writable = data_dir.is_dir() and os.access(data_dir, os.R_OK | os.W_OK | os.X_OK)
    notebooks_dir = settings.notebooks_dir
    notebooks_ok = not notebooks_dir.exists() or os.access(notebooks_dir, os.W_OK)
    health["data"] = {
        "status": "ok" if writable and notebooks_ok else "error",
        "path": str(data_dir),
    }

    try:
        if conn is None:
            raise RuntimeError("settings unavailable: database error")
        effective = settings_service(request, conn).effective()
        report = check_connection(
            effective, transport=llm_transport(request), timeout_s=HEALTH_LLM_TIMEOUT_S
        )
        health["llm"] = {
            "status": "reachable" if report.reachable else "unreachable",
            "base_url": effective.llm_base_url,
            "model": effective.llm_model,
            "model_available": report.model_found,
            "message": report.message,
            "elapsed_ms": round(report.elapsed_ms, 1),
        }
    except Exception as e:  # health must never raise
        health["llm"] = {"status": "unreachable", "message": str(e)}
    finally:
        if conn is not None:
            conn.close()

    embedder = getattr(request.app.state, "embedder", None)
    health["embedding_worker"] = (
        embedder.status() if embedder is not None else {"status": "stopped"}
    )  # never starts the worker

    core_ok = health["database"]["status"] == "ok" and health["data"]["status"] == "ok"
    health["status"] = "ok" if core_ok else "error"
    return health


@router.get("/healthz")
def healthz(request: Request) -> JSONResponse:
    health = collect_health(request)
    return JSONResponse(health, status_code=200 if health["status"] == "ok" else 503)


@router.get("/health/badge", response_class=HTMLResponse)
def health_badge(request: Request) -> HTMLResponse:
    return render(
        request, "partials/health_badge.html", {"health": collect_health(request)}
    )
