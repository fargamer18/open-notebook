"""Request-scoped helpers shared by the route modules."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from litenotebook.config import Settings
from litenotebook.db import connect
from litenotebook.services.settings import SettingsService

TEMPLATES = Jinja2Templates(directory=Path(__file__).parent / "templates")


def filesize(num: int | None) -> str:
    if num is None:
        return "–"
    size = float(num)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds // 60:.0f} min {seconds % 60:.0f} s"


def timestamp(value: str | None) -> str:
    return (value or "")[:19].replace("T", " ") or "–"


TEMPLATES.env.filters.update(filesize=filesize, duration=duration, ts=timestamp)


def get_db(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(app_settings(request).db_path)
    try:
        yield conn
    finally:
        conn.close()


def app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def settings_service(request: Request, conn: sqlite3.Connection) -> SettingsService:
    env: Mapping[str, str] | None = request.app.state.env
    return SettingsService(conn, env)


def llm_transport(request: Request) -> httpx.BaseTransport | None:
    transport: httpx.BaseTransport | None = request.app.state.llm_transport
    return transport


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def render(
    request: Request,
    template: str,
    context: dict[str, Any] | None = None,
    *,
    status_code: int = 200,
) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request, template, context or {}, status_code=status_code
    )


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303)
