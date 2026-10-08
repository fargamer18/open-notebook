"""Notes: list, editor with autosave, links to evidence, "Add to note".

Every route is scoped by the notebook id in the URL. Nothing here starts the
embedding worker or calls the LLM; graph memory is updated deterministically.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from litenotebook.notebooks import get_summary
from litenotebook.notes import (
    MAX_TITLE_CHARS,
    NoteConflictError,
    NoteError,
    NoteNotFoundError,
    append_to_note,
    create_note,
    delete_note,
    get_note,
    link_annotation,
    link_chunk,
    link_citation,
    link_memory_node,
    link_message,
    list_links,
    list_notes,
    note_concepts,
    note_title_for_answer,
    remove_link,
    update_note,
)
from litenotebook.unified_search import (
    clean_query,
    search_notes,
    search_sources_lexical,
)
from litenotebook.web.deps import get_db, is_htmx, redirect, render
from litenotebook.web.navigation import last_read

router = APIRouter(prefix="/notebooks/{notebook_id}/notes")
Db = Annotated[sqlite3.Connection, Depends(get_db)]
ADD_KINDS = ("chunk", "citation", "message", "memory", "annotation")


def note_url(notebook_id: int, note_id: int | None = None) -> str:
    base = f"/notebooks/{notebook_id}/notes"
    return base if note_id is None else f"{base}/{note_id}"


def _page(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    note_id: int | None,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    from litenotebook.web.routes.notebooks import WORKSPACE_TABS

    summary = get_summary(conn, notebook_id)
    query = clean_query(str(extra.pop("q", "") or ""))
    notes = list_notes(conn, notebook_id)
    if query:
        hits = {h.meta["note_id"] for h in search_notes(conn, notebook_id, query, 200)}
        notes = [n for n in notes if n.id in hits]
    note = get_note(conn, notebook_id, note_id) if note_id is not None else None
    return render(
        request,
        "notebook/workspace.html",
        {
            "s": summary,
            "tabs": WORKSPACE_TABS,
            "last_read": last_read(conn, notebook_id),
            "tab": "notes",
            "notes": notes,
            "note": note,
            "links": list_links(conn, notebook_id, note.id) if note else [],
            "concepts": note_concepts(conn, notebook_id, note.id) if note else [],
            "q": query,
            "max_title": MAX_TITLE_CHARS,
            "nb_id": notebook_id,
            **extra,
        },
        status_code=status_code,
    )


@router.get("", response_class=HTMLResponse)
def notes_home(
    request: Request, conn: Db, notebook_id: int, q: str = ""
) -> HTMLResponse:
    return _page(request, conn, notebook_id, None, q=q)


@router.post("", response_model=None)
def new_note(
    request: Request,
    conn: Db,
    notebook_id: int,
    title: Annotated[str, Form()] = "",
    content: Annotated[str, Form()] = "",
) -> Response:
    get_summary(conn, notebook_id)
    try:
        note = create_note(conn, notebook_id, title or "Untitled note", content)
    except NoteError as e:
        return _page(
            request, conn, notebook_id, None, create_error=str(e), status_code=422
        )
    return redirect(note_url(notebook_id, note.id), flash="note-created")


@router.get("/add-form", response_class=HTMLResponse)
def add_form(
    request: Request,
    conn: Db,
    notebook_id: int,
    kind: str = "",
    id: int = 0,  # noqa: A002
) -> HTMLResponse:
    """Small chooser: add an item to a new or an existing note."""
    if kind not in ADD_KINDS:
        raise NoteNotFoundError("Nothing to add")
    return render(
        request,
        "partials/add_to_note_form.html",
        {
            "nb_id": notebook_id,
            "kind": kind,
            "item_id": id,
            "notes": list_notes(conn, notebook_id)[:50],
        },
    )


def _add(
    conn: sqlite3.Connection,
    notebook_id: int,
    kind: str,
    item_id: int,
    note_id: int | None,
) -> int:
    """Create (if needed) and link. Titles are deterministic — never generated."""
    from litenotebook.chat import get_citation, get_message

    if kind == "message":
        message = get_message(conn, notebook_id, item_id)
        if message.role != "assistant" or not message.content.strip():
            raise NoteError("Only answers with text can be saved to a note")
        question = get_message(conn, notebook_id, message.reply_to or 0).content
        text = message.content
        if message.citations:
            text += "\n\nSources:\n" + "\n".join(
                f"[S{c.label}] {c.source_filename} — page {c.page_no}, "
                f"chunk {c.chunk_ord}"
                for c in message.citations
            )
        if note_id is None:
            note_id = create_note(
                conn, notebook_id, note_title_for_answer(question), text
            ).id
        else:
            append_to_note(conn, notebook_id, note_id, text)
        link_message(conn, notebook_id, note_id, message.id)
        for c in message.citations:
            link_citation(conn, notebook_id, note_id, c.id)
        return note_id
    if note_id is None:
        if kind == "chunk":
            row = conn.execute(
                "SELECT s.original_filename, c.page_no FROM chunks c "
                "JOIN sources s ON s.id = c.source_id "
                "WHERE c.id = ? AND c.notebook_id = ?",
                (item_id, notebook_id),
            ).fetchone()
            if row is None:
                raise NoteNotFoundError("That passage is not in this notebook")
            title = f"Evidence: {row[0]} p.{row[1]}"
        elif kind == "citation":
            c = get_citation(conn, notebook_id, item_id)
            title = f"Evidence: {c.source_filename} p.{c.page_no}"
        elif kind == "annotation":
            from litenotebook.reader.annotations import get_annotation

            a = get_annotation(conn, notebook_id, item_id)
            title = f"Annotation: {a.source_filename} p.{a.page_no}"
        else:
            row = conn.execute(
                "SELECT label FROM memory_nodes WHERE id = ? AND notebook_id = ?",
                (item_id, notebook_id),
            ).fetchone()
            if row is None:
                raise NoteNotFoundError("That memory item is not in this notebook")
            title = f"Memory: {row[0]}"
        note_id = create_note(conn, notebook_id, title[:MAX_TITLE_CHARS]).id
    {
        "chunk": link_chunk,
        "citation": link_citation,
        "memory": link_memory_node,
        "annotation": link_annotation,
    }[kind](conn, notebook_id, note_id, item_id)
    return note_id


@router.post("/add", response_model=None)
def add(
    request: Request,
    conn: Db,
    notebook_id: int,
    kind: Annotated[str, Form()] = "",
    item_id: Annotated[int, Form()] = 0,
    note_id: Annotated[str, Form()] = "new",
) -> Response:
    if kind not in ADD_KINDS:
        return JSONResponse({"error": "Nothing to add"}, status_code=422)
    target = None if note_id in ("", "new") else int(note_id)
    if target is not None:
        get_note(conn, notebook_id, target)  # 404 for another notebook's note
    try:
        saved = _add(conn, notebook_id, kind, item_id, target)
    except NoteNotFoundError:
        raise
    except NoteError as e:
        if is_htmx(request):
            return render(
                request,
                "partials/add_to_note_done.html",
                {"error": str(e)},
                status_code=422,
            )
        return JSONResponse({"error": str(e)}, status_code=422)
    if not is_htmx(request):
        return redirect(note_url(notebook_id, saved))
    return render(
        request,
        "partials/add_to_note_done.html",
        {"note": get_note(conn, notebook_id, saved), "nb_id": notebook_id},
    )


@router.get("/{note_id}", response_class=HTMLResponse)
def open_note(
    request: Request, conn: Db, notebook_id: int, note_id: int
) -> HTMLResponse:
    return _page(request, conn, notebook_id, note_id)


@router.post("/{note_id}/autosave", response_model=None)
def autosave(
    conn: Db,
    notebook_id: int,
    note_id: int,
    title: Annotated[str, Form()] = "",
    content: Annotated[str, Form()] = "",
    revision: Annotated[int, Form()] = 0,
) -> JSONResponse:
    """JSON for notes.js; a stale revision is a 409 instead of an overwrite."""
    try:
        note = update_note(
            conn,
            notebook_id,
            note_id,
            title=title,
            content=content,
            base_revision=revision,
        )
    except NoteConflictError as e:
        return JSONResponse({"ok": False, "error": str(e), "conflict": True}, 409)
    except NoteNotFoundError:
        raise
    except NoteError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)
    return JSONResponse(
        {
            "ok": True,
            "revision": note.revision,
            "saved_at": note.updated_at,
            "title": note.title,
        }
    )


@router.post("/{note_id}", response_model=None)
def save_note(
    request: Request,
    conn: Db,
    notebook_id: int,
    note_id: int,
    title: Annotated[str, Form()] = "",
    content: Annotated[str, Form()] = "",
    revision: Annotated[int, Form()] = 0,
) -> Response:
    """Fallback for a plain form submit (no JavaScript)."""
    try:
        update_note(
            conn,
            notebook_id,
            note_id,
            title=title,
            content=content,
            base_revision=revision or None,
        )
    except NoteNotFoundError:
        raise
    except NoteError as e:
        return _page(
            request,
            conn,
            notebook_id,
            note_id,
            save_error=str(e),
            draft_title=title,
            draft_content=content,
            status_code=422,
        )
    return redirect(note_url(notebook_id, note_id))


@router.get("/{note_id}/delete", response_class=HTMLResponse)
def delete_confirm(
    request: Request, conn: Db, notebook_id: int, note_id: int
) -> HTMLResponse:
    return _page(request, conn, notebook_id, note_id, confirming=True)


@router.post("/{note_id}/delete", response_model=None)
def delete(
    request: Request,
    conn: Db,
    notebook_id: int,
    note_id: int,
    confirm: Annotated[str, Form()] = "",
) -> Response:
    get_note(conn, notebook_id, note_id)
    if confirm != str(note_id):
        return _page(
            request,
            conn,
            notebook_id,
            note_id,
            confirming=True,
            delete_error="Deletion not confirmed.",
            status_code=422,
        )
    delete_note(conn, notebook_id, note_id)
    return redirect(note_url(notebook_id), flash="note-deleted")


@router.get("/{note_id}/attach", response_class=HTMLResponse)
def attach_search(
    request: Request, conn: Db, notebook_id: int, note_id: int, q: str = ""
) -> HTMLResponse:
    """Find passages to attach (lexical only: never starts the embedding model)."""
    get_note(conn, notebook_id, note_id)
    query = clean_query(q)
    hits = search_sources_lexical(conn, notebook_id, query, 10) if query else []
    return render(
        request,
        "partials/note_attach_results.html",
        {"hits": hits, "q": query, "nb_id": notebook_id, "note_id": note_id},
    )


@router.post("/{note_id}/links", response_model=None)
def attach(
    request: Request,
    conn: Db,
    notebook_id: int,
    note_id: int,
    kind: Annotated[str, Form()] = "",
    item_id: Annotated[int, Form()] = 0,
) -> Response:
    get_note(conn, notebook_id, note_id)
    if kind not in ADD_KINDS:
        return JSONResponse({"error": "Unknown link type"}, status_code=422)
    try:
        _add(conn, notebook_id, kind, item_id, note_id)
    except NoteNotFoundError:
        raise
    except NoteError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    return _links(request, conn, notebook_id, note_id)


@router.post("/{note_id}/links/{link_id}/delete", response_model=None)
def unlink(
    request: Request, conn: Db, notebook_id: int, note_id: int, link_id: int
) -> Response:
    remove_link(conn, notebook_id, note_id, link_id)
    if not is_htmx(request):
        return redirect(note_url(notebook_id, note_id))
    return _links(request, conn, notebook_id, note_id)


def _links(
    request: Request, conn: sqlite3.Connection, notebook_id: int, note_id: int
) -> HTMLResponse:
    return render(
        request,
        "partials/note_links.html",
        {
            "note": get_note(conn, notebook_id, note_id),
            "links": list_links(conn, notebook_id, note_id),
            "concepts": note_concepts(conn, notebook_id, note_id),
            "nb_id": notebook_id,
        },
    )
