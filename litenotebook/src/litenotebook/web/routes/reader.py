"""Built-in PDF reader: page images, text geometry, annotations, reading state.

Every route is scoped by the notebook (and source) ids in the URL. Nothing
here starts the embedding worker or calls the LLM; PDFs are rendered
passively by PyMuPDF and never modified.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from litenotebook.notebooks import get_summary
from litenotebook.reader.annotations import (
    COLORS,
    AnnotationError,
    AnnotationNotFoundError,
    chunk_target,
    citation_target,
    create_annotation,
    delete_annotation,
    get_annotation,
    get_reading_state,
    list_annotations,
    save_reading_state,
    update_annotation,
)
from litenotebook.reader.render import (
    PageNotFoundError,
    ReaderError,
    check_page,
    open_source,
    render_page,
)
from litenotebook.web.deps import app_settings, get_db, is_htmx, render
from litenotebook.web.navigation import notify

router = APIRouter(prefix="/notebooks/{notebook_id}/sources/{source_id}")
panels = APIRouter(prefix="/notebooks/{notebook_id}/reader")
Db = Annotated[sqlite3.Connection, Depends(get_db)]


def reader_url(notebook_id: int, source_id: int, **params: int) -> str:
    query = "&".join(f"{k}={int(v)}" for k, v in params.items())
    return f"/notebooks/{notebook_id}/sources/{source_id}/read" + (
        f"?{query}" if query else ""
    )


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@router.get("/read", response_class=HTMLResponse)
def reader_page(
    request: Request,
    conn: Db,
    notebook_id: int,
    source_id: int,
    page: int | None = None,
    chunk: int | None = None,
    citation: int | None = None,
    annotation: int | None = None,
) -> HTMLResponse:
    settings = app_settings(request)
    summary = get_summary(conn, notebook_id)
    try:
        reader = open_source(conn, settings, notebook_id, source_id)
    except ReaderError as e:
        return render(
            request,
            "error.html",
            {
                "title": "PDF unavailable",
                "message": str(e),
                "status": 404,
                "notebook_id": notebook_id,
            },
            status_code=404,
        )
    saved_page, zoom = get_reading_state(conn, notebook_id, source_id)
    start, focus_kind, focus_id = saved_page, "", 0
    back_url = None  # "Back to chat" when the reader was opened from a citation
    if page is not None:
        start = page
    if chunk is not None:
        row = conn.execute(
            "SELECT page_no FROM chunks WHERE id = ? AND source_id = ? "
            "AND notebook_id = ?",
            (chunk, source_id, notebook_id),
        ).fetchone()
        if row is not None:  # a citation/link overrides the saved position
            start, focus_kind, focus_id = row[0], "chunk", chunk
    if citation is not None:
        from litenotebook.chat import get_citation

        cit = get_citation(conn, notebook_id, citation)
        if cit.chunk_exists and cit.source_id == source_id:
            start, focus_kind, focus_id = cit.page_no, "citation", citation
        row = conn.execute(
            "SELECT conversation_id FROM messages WHERE id = ?", (cit.message_id,)
        ).fetchone()
        if row is not None:
            back_url = f"/notebooks/{notebook_id}/chat/{row[0]}#msg-{cit.message_id}"
    if annotation is not None:
        a = get_annotation(conn, notebook_id, annotation)
        if a.source_id != source_id:
            raise AnnotationNotFoundError("That annotation belongs to another source")
        start, focus_kind, focus_id = a.page_no, "annotation", a.id
    start = min(max(1, start), reader.page_count)
    return render(
        request,
        "notebook/reader.html",
        {
            "s": summary,
            "src": reader.source,
            "page_count": reader.page_count,
            "start_page": start,
            "zoom": zoom,
            "focus_kind": focus_kind,
            "focus_id": focus_id,
            "colors": COLORS,
            "nb_id": notebook_id,
            "back_url": back_url,
        },
    )


@router.get("/pages/{page_no}/render", response_model=None)
def page_image(
    request: Request,
    conn: Db,
    notebook_id: int,
    source_id: int,
    page_no: int,
    scale: float = 1.5,
) -> Response:
    settings = app_settings(request)
    try:
        reader = open_source(conn, settings, notebook_id, source_id)
        path, cached = render_page(settings, reader, page_no, scale)
    except PageNotFoundError as e:
        return _error(str(e), 404)
    except ReaderError as e:
        return _error(str(e), 404)
    return FileResponse(
        path,
        media_type="image/png",
        headers={
            "Cache-Control": "private, max-age=604800",
            "X-Render-Cache": "hit" if cached else "miss",
        },
    )


@router.get("/pages/{page_no}/text", response_model=None)
def page_text(
    request: Request, conn: Db, notebook_id: int, source_id: int, page_no: int
) -> JSONResponse:
    from litenotebook.reader.geometry import page_geometry

    settings = app_settings(request)
    try:
        reader = open_source(conn, settings, notebook_id, source_id)
        page_no = check_page(reader, page_no)
    except ReaderError as e:
        return _error(str(e), 404)
    geometry = page_geometry(reader.path, page_no)
    method = conn.execute(
        "SELECT method FROM pages WHERE source_id = ? AND page_no = ?",
        (source_id, page_no),
    ).fetchone()
    return JSONResponse(
        {
            "page": page_no,
            "has_text": geometry.has_text,
            "method": method[0] if method else None,
            "words": [w.as_dict() for w in geometry.words],
        }
    )


@router.get("/annotations.json", response_model=None)
def annotations_json(
    conn: Db, notebook_id: int, source_id: int, page: int | None = None
) -> JSONResponse:
    get_summary(conn, notebook_id)
    items = list_annotations(conn, notebook_id, source_id, page)
    return JSONResponse({"annotations": [a.as_dict() for a in items]})


@router.get("/annotations", response_class=HTMLResponse)
def annotations_panel(
    request: Request, conn: Db, notebook_id: int, source_id: int
) -> HTMLResponse:
    from litenotebook.sources import get_source

    source = get_source(conn, notebook_id, source_id)
    return render(
        request,
        "partials/reader_annotations.html",
        {
            "items": list_annotations(conn, notebook_id, source_id),
            "src": source,
            "nb_id": notebook_id,
            "colors": COLORS,
        },
    )


@router.post("/annotations", response_model=None)
def create(
    request: Request,
    conn: Db,
    notebook_id: int,
    source_id: int,
    page: Annotated[int, Form()] = 0,
    kind: Annotated[str, Form()] = "highlight",
    text: Annotated[str, Form()] = "",
    comment: Annotated[str, Form()] = "",
    color: Annotated[str, Form()] = "yellow",
    rects: Annotated[str, Form()] = "[]",
) -> JSONResponse:
    try:
        parsed: Any = json.loads(rects or "[]")
    except ValueError:
        return _error("Invalid selection", 422)
    try:
        a = create_annotation(
            conn,
            app_settings(request),
            notebook_id,
            source_id,
            page_no=page,
            annotation_type=kind,
            selected_text=text,
            comment=comment,
            color=color,
            rects=parsed,
        )
    except PageNotFoundError as e:
        return _error(str(e), 404)
    except AnnotationError as e:
        return _error(str(e), 422)
    except ReaderError as e:
        return _error(str(e), 404)
    return JSONResponse(a.as_dict(), status_code=201)


def _annotation_for_source(
    conn: sqlite3.Connection, notebook_id: int, source_id: int, annotation_id: int
) -> Any:
    a = get_annotation(conn, notebook_id, annotation_id)
    if a.source_id != source_id:
        raise AnnotationNotFoundError("Annotation not found for this source")
    return a


@router.get("/annotations/{annotation_id}/edit", response_class=HTMLResponse)
def edit_form(
    request: Request, conn: Db, notebook_id: int, source_id: int, annotation_id: int
) -> HTMLResponse:
    a = _annotation_for_source(conn, notebook_id, source_id, annotation_id)
    return render(
        request,
        "partials/reader_annotation.html",
        {"a": a, "mode": "edit", "nb_id": notebook_id, "colors": COLORS},
    )


@router.post("/annotations/{annotation_id}", response_model=None)
def edit(
    request: Request,
    conn: Db,
    notebook_id: int,
    source_id: int,
    annotation_id: int,
    comment: Annotated[str | None, Form()] = None,
    color: Annotated[str | None, Form()] = None,
) -> Response:
    _annotation_for_source(conn, notebook_id, source_id, annotation_id)
    try:
        a = update_annotation(
            conn, notebook_id, annotation_id, comment=comment, color=color
        )
    except AnnotationError as e:
        if is_htmx(request):
            current = get_annotation(conn, notebook_id, annotation_id)
            return render(
                request,
                "partials/reader_annotation.html",
                {
                    "a": current,
                    "mode": "edit",
                    "error": str(e),
                    "nb_id": notebook_id,
                    "colors": COLORS,
                },
                status_code=422,
            )
        return _error(str(e), 422)
    if not is_htmx(request):
        return JSONResponse(a.as_dict())
    response = render(
        request,
        "partials/reader_annotation.html",
        {"a": a, "mode": "view", "nb_id": notebook_id, "colors": COLORS},
    )
    response.headers["HX-Trigger"] = "annotations-changed"
    return notify(response, "Annotation updated")


@router.get("/annotations/{annotation_id}/delete", response_class=HTMLResponse)
def delete_confirm(
    request: Request, conn: Db, notebook_id: int, source_id: int, annotation_id: int
) -> HTMLResponse:
    a = _annotation_for_source(conn, notebook_id, source_id, annotation_id)
    return render(
        request,
        "partials/reader_annotation.html",
        {"a": a, "mode": "confirm", "nb_id": notebook_id, "colors": COLORS},
    )


@router.post("/annotations/{annotation_id}/delete", response_model=None)
def delete(
    request: Request,
    conn: Db,
    notebook_id: int,
    source_id: int,
    annotation_id: int,
    confirm: Annotated[str, Form()] = "",
) -> Response:
    a = _annotation_for_source(conn, notebook_id, source_id, annotation_id)
    if confirm != str(annotation_id):
        return render(
            request,
            "partials/reader_annotation.html",
            {
                "a": a,
                "mode": "confirm",
                "error": "Deletion not confirmed.",
                "nb_id": notebook_id,
                "colors": COLORS,
            },
            status_code=422,
        )
    delete_annotation(conn, notebook_id, annotation_id)
    response = Response(status_code=200, content="")
    response.headers["HX-Trigger"] = "annotations-changed"
    return notify(response, "Annotation deleted")


@router.post("/reading-state", response_model=None)
def reading_state(
    request: Request,
    conn: Db,
    notebook_id: int,
    source_id: int,
    page: Annotated[int, Form()] = 1,
    zoom: Annotated[float, Form()] = 1.0,
) -> JSONResponse:
    try:
        reader = open_source(conn, app_settings(request), notebook_id, source_id)
        saved_page, saved_zoom = save_reading_state(conn, reader, page, zoom)
    except ReaderError as e:
        return _error(str(e), 404 if isinstance(e, PageNotFoundError) else 422)
    return JSONResponse({"page": saved_page, "zoom": saved_zoom})


@router.get("/chunks/{chunk_id}/highlight", response_model=None)
def chunk_highlight(
    request: Request, conn: Db, notebook_id: int, source_id: int, chunk_id: int
) -> JSONResponse:
    try:
        target = chunk_target(conn, app_settings(request), notebook_id, chunk_id)
    except ReaderError as e:
        return _error(str(e), 404)
    if target.source_id != source_id:
        return _error("That passage belongs to another source", 404)
    return JSONResponse(_target_json(target))


def _target_json(target: Any) -> dict[str, Any]:
    return {
        "source_id": target.source_id,
        "page": target.page_no,
        "rects": target.rects,
        "live": target.live,
        "reason": target.reason,
        "text": target.text,
    }


@panels.get("/citations/{citation_id}", response_model=None)
def citation_highlight(
    request: Request, conn: Db, notebook_id: int, citation_id: int
) -> JSONResponse:
    try:
        target = citation_target(conn, app_settings(request), notebook_id, citation_id)
    except ReaderError as e:
        return _error(str(e), 404)
    return JSONResponse(_target_json(target))


@panels.get("/panel/notes", response_class=HTMLResponse)
def notes_panel(
    request: Request, conn: Db, notebook_id: int, note: int | None = None
) -> HTMLResponse:
    from litenotebook.notes import get_note, list_links, list_notes, note_concepts

    get_summary(conn, notebook_id)
    current = get_note(conn, notebook_id, note) if note else None
    return render(
        request,
        "partials/reader_notes.html",
        {
            "notes": list_notes(conn, notebook_id),
            "note": current,
            "links": list_links(conn, notebook_id, current.id) if current else [],
            "concepts": note_concepts(conn, notebook_id, current.id) if current else [],
            "nb_id": notebook_id,
        },
    )


@panels.get("/panel/chat", response_class=HTMLResponse)
def chat_panel(
    request: Request, conn: Db, notebook_id: int, c: int | None = None
) -> HTMLResponse:
    from litenotebook.chat import (
        MAX_QUESTION_CHARS,
        MAX_TITLE_CHARS,
        get_conversation,
        list_conversations,
        list_messages,
        reconcile_generating,
    )

    summary = get_summary(conn, notebook_id)
    reconcile_generating(conn, request.app.state.generation_guard.active)
    conversations = list_conversations(conn, notebook_id)
    if c == 0:  # "New chat"
        current = None
    elif c is not None:
        current = get_conversation(conn, notebook_id, c)
    else:
        current = conversations[0] if conversations else None
    ready = conn.execute(
        "SELECT count(*) FROM sources WHERE notebook_id = ? AND status = 'ready'",
        (notebook_id,),
    ).fetchone()[0]
    return render(
        request,
        "partials/reader_chat.html",
        {
            "s": summary,
            "conversations": conversations,
            "conversation": current,
            "messages": list_messages(conn, current.id) if current else [],
            "ready_sources": ready,
            "max_question": MAX_QUESTION_CHARS,
            "max_title": MAX_TITLE_CHARS,
            "in_reader": True,
        },
    )
