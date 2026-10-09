"""Shell helpers: notebook navigation context, flash messages and toasts.

Nothing here touches the embedding model or the LLM; it only reads SQLite.

Feedback reaches the browser in two ways:

- HTMX responses carry ``X-LN-Notify: {"kind": ..., "message": ...}``; the
  shared script reads it in ``htmx:beforeSwap`` (fired on the target, which
  is still attached, unlike a requesting element that the swap removes).
- Full-page redirects set a short-lived ``ln_flash`` cookie holding a *key*
  from ``FLASH_MESSAGES``, never free text, so nothing user-controlled is
  echoed from a cookie. The next full page shows the message once.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from starlette.responses import Response

FLASH_COOKIE = "ln_flash"
NOTIFY_HEADER = "X-LN-Notify"
FLASH_MESSAGES: dict[str, tuple[str, str]] = {
    "notebook-deleted": ("success", "Notebook deleted"),
    "source-deleted": ("success", "Source deleted"),
    "conversation-deleted": ("success", "Conversation deleted"),
    "note-created": ("success", "Note created"),
    "note-deleted": ("success", "Note deleted"),
    "settings-saved": ("success", "Settings saved"),
    "restore-cancelled": ("info", "Pending restore cancelled"),
    "ai-updated": ("success", "Local AI updated"),
    "quiz-deleted": ("success", "Quiz deleted"),
}


def last_read(conn: sqlite3.Connection, notebook_id: int) -> dict[str, Any] | None:
    """The PDF most recently open in the reader, if it is still ready."""
    row = conn.execute(
        "SELECT s.id, s.original_filename FROM reading_state r "
        "JOIN sources s ON s.id = r.source_id "
        "WHERE r.notebook_id = ? AND s.notebook_id = ? AND s.status = 'ready' "
        "ORDER BY r.updated_at DESC LIMIT 1",
        (notebook_id, notebook_id),
    ).fetchone()
    return {"id": row[0], "filename": row[1]} if row else None


def nav_context(conn: sqlite3.Connection, notebook_id: int) -> dict[str, Any]:
    return {"last_read": last_read(conn, notebook_id)}


def set_flash(response: Response, key: str) -> Response:
    if key not in FLASH_MESSAGES:
        raise ValueError(f"unknown flash message {key!r}")
    response.set_cookie(
        FLASH_COOKIE, key, max_age=60, path="/", httponly=True, samesite="strict"
    )
    return response


def read_flash(cookies: dict[str, str]) -> dict[str, str] | None:
    entry = FLASH_MESSAGES.get(cookies.get(FLASH_COOKIE, ""))
    return {"kind": entry[0], "message": entry[1]} if entry else None


def notify(response: Response, message: str, kind: str = "success") -> Response:
    """Ask the page to show a toast for this HTMX response."""
    payload = {"kind": kind, "message": message[:200]}
    # ensure_ascii keeps the header Latin-1 safe; the toast uses textContent.
    response.headers[NOTIFY_HEADER] = json.dumps(payload)
    return response
