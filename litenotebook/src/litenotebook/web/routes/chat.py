"""Notebook chat: conversations, streamed answers, retry/stop and evidence.

Every route is scoped by the notebook id in the URL; conversation, message and
citation ids from another notebook are "not found". Rendering pages and
history never imports retrieval/MiniLM; only sending a question does (inside
the generation, in a worker thread).
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from httpx import AsyncBaseTransport
from starlette.concurrency import run_in_threadpool

from litenotebook.chat import (
    INTERRUPTED_MESSAGE as _INTERRUPTED,
)
from litenotebook.chat import (
    MAX_QUESTION_CHARS,
    MAX_TITLE_CHARS,
    ChatBusyError,
    ChatError,
    CitationNotFoundError,
    ConversationNotFoundError,
    Exchange,
    MessageNotFoundError,
    clean_question,
    delete_conversation,
    end_message,
    get_chunk_evidence,
    get_citation,
    get_conversation,
    get_message,
    list_conversations,
    list_messages,
    reconcile_generating,
    rename_conversation,
    start_exchange,
    start_retry,
)
from litenotebook.db import connect
from litenotebook.notebooks import get_notebook, get_summary
from litenotebook.services.chat import BUSY_MESSAGE, ChatGeneration, GenerationGuard
from litenotebook.web.deps import (
    app_settings,
    get_db,
    is_htmx,
    llm_transport,
    redirect,
    render,
    settings_service,
)
from litenotebook.web.streaming import EventStream

router = APIRouter(prefix="/notebooks/{notebook_id}/chat")
Db = Annotated[sqlite3.Connection, Depends(get_db)]

NOT_FOUND = (ConversationNotFoundError, MessageNotFoundError, CitationNotFoundError)


def guard(request: Request) -> GenerationGuard:
    g: GenerationGuard = request.app.state.generation_guard
    return g


def chat_url(notebook_id: int, conversation_id: int | None = None) -> str:
    base = f"/notebooks/{notebook_id}/chat"
    return base if conversation_id is None else f"{base}/{conversation_id}"


def _ready_sources(conn: sqlite3.Connection, notebook_id: int) -> int:
    row = conn.execute(
        "SELECT count(*) FROM sources WHERE notebook_id = ? AND status = 'ready'",
        (notebook_id,),
    ).fetchone()
    return int(row[0])


def _page(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    conversation_id: int | None,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    from litenotebook.web.routes.notebooks import WORKSPACE_TABS

    reconcile_generating(conn, guard(request).active)
    conversation = (
        get_conversation(conn, notebook_id, conversation_id)
        if conversation_id is not None
        else None
    )
    return render(
        request,
        "notebook/workspace.html",
        {
            "s": get_summary(conn, notebook_id),
            "tabs": WORKSPACE_TABS,
            "tab": "chat",
            "conversations": list_conversations(conn, notebook_id),
            "conversation": conversation,
            "messages": list_messages(conn, conversation.id) if conversation else [],
            "ready_sources": _ready_sources(conn, notebook_id),
            "max_question": MAX_QUESTION_CHARS,
            "max_title": MAX_TITLE_CHARS,
            **extra,
        },
        status_code=status_code,
    )


# --- pages ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
def chat_home(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    """The most recent conversation, or the "New chat" state if there is none."""
    get_notebook(conn, notebook_id)
    conversations = list_conversations(conn, notebook_id)
    latest = conversations[0].id if conversations else None
    return _page(request, conn, notebook_id, latest)


@router.get("/new", response_class=HTMLResponse)
def new_chat(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    get_notebook(conn, notebook_id)
    return _page(request, conn, notebook_id, None)


@router.get("/sidebar", response_class=HTMLResponse)
def sidebar(
    request: Request, conn: Db, notebook_id: int, current: int | None = None
) -> HTMLResponse:
    get_notebook(conn, notebook_id)
    return render(
        request,
        "partials/chat_conversations.html",
        {
            "s": get_summary(conn, notebook_id),
            "conversations": list_conversations(conn, notebook_id),
            "current_id": current,
        },
    )


@router.get("/messages/{message_id}", response_class=HTMLResponse)
def message_fragment(
    request: Request, conn: Db, notebook_id: int, message_id: int
) -> HTMLResponse:
    reconcile_generating(conn, guard(request).active)
    return render(
        request,
        "partials/chat_message.html",
        {"m": get_message(conn, notebook_id, message_id), "nb_id": notebook_id},
    )


@router.get("/citations/{citation_id}", response_class=HTMLResponse)
def citation_evidence(
    request: Request, conn: Db, notebook_id: int, citation_id: int
) -> HTMLResponse:
    citation = get_citation(conn, notebook_id, citation_id)
    live = (
        get_chunk_evidence(conn, notebook_id, citation.chunk_id)
        if citation.chunk_exists
        else None
    )
    context = {"citation": citation, "live": live, "nb_id": notebook_id}
    if is_htmx(request):
        return render(request, "partials/chat_evidence.html", context)
    return render(
        request,
        "notebook/evidence.html",
        {**context, "s": get_summary(conn, notebook_id)},
    )


@router.get("/chunks/{chunk_id}", response_class=HTMLResponse)
def chunk_evidence(
    request: Request, conn: Db, notebook_id: int, chunk_id: int
) -> HTMLResponse:
    """Previous/next navigation: live chunks of ready sources in this notebook."""
    live = get_chunk_evidence(conn, notebook_id, chunk_id)
    if live is None:
        raise CitationNotFoundError(f"Chunk {chunk_id} not found in this notebook")
    context = {"citation": None, "live": live, "nb_id": notebook_id}
    if is_htmx(request):
        return render(request, "partials/chat_evidence.html", context)
    return render(
        request,
        "notebook/evidence.html",
        {**context, "s": get_summary(conn, notebook_id)},
    )


# --- streaming: send / retry / stop -------------------------------------------------


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _stream(request: Request, notebook_id: int, exchange: Exchange) -> EventStream:
    settings = app_settings(request)
    g = guard(request)
    g.bind(exchange.assistant_message_id)
    conn = connect(settings.db_path)
    try:
        llm = settings_service(request, conn).effective()
    finally:
        conn.close()
    transport = llm_transport(request)
    generation = ChatGeneration(
        settings=settings,
        llm=llm,
        notebook_id=notebook_id,
        exchange=exchange,
        embedder=request.app.state.embedder,
        guard=g,
        # tests inject httpx.MockTransport, which is both sync and async
        transport=transport if isinstance(transport, AsyncBaseTransport) else None,
    )
    message_id = exchange.assistant_message_id

    def on_close() -> None:
        g.release(message_id)
        cleanup = connect(settings.db_path)
        try:
            end_message(cleanup, message_id, "interrupted", _INTERRUPTED)
        finally:
            cleanup.close()

    return EventStream(generation.events(), on_close=on_close)


def _begin(
    request: Request, notebook_id: int, start: Any, *args: Any
) -> Exchange | JSONResponse:
    """Take the global generation lease, then persist the exchange."""
    g = guard(request)
    if not g.try_acquire():
        return _error(BUSY_MESSAGE, 409)
    conn = connect(app_settings(request).db_path)
    try:
        reconcile_generating(conn, set())  # nothing else can be generating now
        exchange: Exchange = start(conn, notebook_id, *args)
        return exchange
    except NOT_FOUND as e:
        g.release()
        return _error(str(e), 404)
    except ChatBusyError as e:
        g.release()
        return _error(str(e), 409)
    except ChatError as e:
        g.release()
        return _error(str(e), 422)
    except BaseException:
        g.release()
        raise
    finally:
        conn.close()


@router.post("/send", response_model=None)
async def send(
    request: Request,
    notebook_id: int,
    question: Annotated[str, Form()] = "",
    conversation_id: Annotated[str, Form()] = "",
) -> Response:
    settings = app_settings(request)

    def check() -> int | None:
        conn = connect(settings.db_path)
        try:
            get_notebook(conn, notebook_id)
            if not conversation_id.strip():
                return None
            cid = int(conversation_id)
            get_conversation(conn, notebook_id, cid)
            return cid
        finally:
            conn.close()

    try:
        clean_question(question)
        cid = await run_in_threadpool(check)
    except ValueError as e:  # ChatError is a ValueError; so is a bad int
        status = 404 if isinstance(e, NOT_FOUND) else 422
        message = str(e) if isinstance(e, ChatError) else "Invalid conversation"
        return _error(message, status)
    result = await run_in_threadpool(
        _begin, request, notebook_id, start_exchange, cid, question
    )
    if isinstance(result, JSONResponse):
        return result
    return _stream(request, notebook_id, result)


@router.post("/messages/{message_id}/retry", response_model=None)
async def retry(request: Request, notebook_id: int, message_id: int) -> Response:
    result = await run_in_threadpool(
        _begin, request, notebook_id, start_retry, message_id
    )
    if isinstance(result, JSONResponse):
        return result
    return _stream(request, notebook_id, result)


def _message_response(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    message_id: int,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    return render(
        request,
        "partials/chat_message.html",
        {
            "m": get_message(conn, notebook_id, message_id),
            "nb_id": notebook_id,
            **extra,
        },
        status_code=status_code,
    )


@router.get("/messages/{message_id}/correct", response_class=HTMLResponse)
def correction_form(
    request: Request, conn: Db, notebook_id: int, message_id: int
) -> Response:
    message = get_message(conn, notebook_id, message_id)
    if not is_htmx(request):
        return redirect(
            f"{chat_url(notebook_id, message.conversation_id)}#msg-{message_id}"
        )
    return _message_response(
        request, conn, notebook_id, message_id, feedback_mode="correct"
    )


@router.post("/messages/{message_id}/outcome", response_model=None)
def outcome(
    request: Request,
    conn: Db,
    notebook_id: int,
    message_id: int,
    outcome: Annotated[str, Form()] = "",
    correction: Annotated[str, Form()] = "",
) -> Response:
    """Useful / dead end / corrected / clear. Never calls the LLM and never
    rewrites the original answer; the memory graph is updated at once."""
    from litenotebook.memory.graph import OutcomeError, set_outcome

    message = get_message(conn, notebook_id, message_id)  # scope check
    choice = {
        "useful": "useful",
        "dead_end": "dead_end",
        "corrected": "corrected",
        "clear": None,
    }
    if outcome not in choice:
        error: str | None = "Choose Useful, Dead end or Correct"
    else:
        error = None
        try:
            set_outcome(conn, notebook_id, message_id, choice[outcome], correction)
        except OutcomeError as e:
            error = str(e)
    if error is not None:
        if is_htmx(request):
            return _message_response(
                request,
                conn,
                notebook_id,
                message_id,
                status_code=422,
                feedback_error=error,
                feedback_mode="correct" if outcome == "corrected" else None,
                correction_value=correction,
            )
        return _error(error, 422)
    if not is_htmx(request):
        return redirect(
            f"{chat_url(notebook_id, message.conversation_id)}#msg-{message_id}"
        )
    return _message_response(request, conn, notebook_id, message_id)


@router.post("/messages/{message_id}/stop", response_model=None)
def stop(request: Request, conn: Db, notebook_id: int, message_id: int) -> Response:
    get_message(conn, notebook_id, message_id)  # scope check
    if guard(request).request_stop(message_id):
        return Response(status_code=204)
    return _error("This answer is not being generated", 409)


# --- rename / delete -----------------------------------------------------------------


def _header(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    conversation_id: int,
    mode: str,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    return render(
        request,
        "partials/chat_header.html",
        {
            "conversation": get_conversation(conn, notebook_id, conversation_id),
            "nb_id": notebook_id,
            "mode": mode,
            "max_title": MAX_TITLE_CHARS,
            **extra,
        },
        status_code=status_code,
    )


@router.get("/{conversation_id}", response_class=HTMLResponse)
def open_conversation(
    request: Request, conn: Db, notebook_id: int, conversation_id: int
) -> HTMLResponse:
    get_notebook(conn, notebook_id)
    return _page(request, conn, notebook_id, conversation_id)


@router.get("/{conversation_id}/header", response_class=HTMLResponse)
def header(
    request: Request, conn: Db, notebook_id: int, conversation_id: int
) -> HTMLResponse:
    return _header(request, conn, notebook_id, conversation_id, "view")


@router.get("/{conversation_id}/rename", response_class=HTMLResponse)
def rename_form(
    request: Request, conn: Db, notebook_id: int, conversation_id: int
) -> HTMLResponse:
    if is_htmx(request):
        return _header(request, conn, notebook_id, conversation_id, "edit")
    return _page(request, conn, notebook_id, conversation_id, header_mode="edit")


@router.post("/{conversation_id}/rename", response_model=None)
def rename(
    request: Request,
    conn: Db,
    notebook_id: int,
    conversation_id: int,
    title: Annotated[str, Form()] = "",
) -> Response:
    try:
        rename_conversation(conn, notebook_id, conversation_id, title)
        from litenotebook.memory.graph import sync_conversation

        sync_conversation(conn, conversation_id)  # memory shows the new title
    except ConversationNotFoundError:
        raise
    except ChatError as e:
        if is_htmx(request):
            return _header(
                request,
                conn,
                notebook_id,
                conversation_id,
                "edit",
                error=str(e),
                value=title,
                status_code=422,
            )
        return _page(
            request,
            conn,
            notebook_id,
            conversation_id,
            header_mode="edit",
            header_error=str(e),
            status_code=422,
        )
    if not is_htmx(request):
        return redirect(chat_url(notebook_id, conversation_id))
    response = _header(request, conn, notebook_id, conversation_id, "view")
    response.headers["HX-Trigger"] = "conversations-changed"
    return response


@router.get("/{conversation_id}/delete", response_class=HTMLResponse)
def delete_confirm(
    request: Request, conn: Db, notebook_id: int, conversation_id: int
) -> HTMLResponse:
    if is_htmx(request):
        return _header(request, conn, notebook_id, conversation_id, "confirm")
    return _page(request, conn, notebook_id, conversation_id, header_mode="confirm")


@router.post("/{conversation_id}/delete", response_model=None)
def delete(
    request: Request,
    conn: Db,
    notebook_id: int,
    conversation_id: int,
    confirm: Annotated[str, Form()] = "",
) -> Response:
    get_conversation(conn, notebook_id, conversation_id)
    error = None
    if confirm != str(conversation_id):
        error, status = "Deletion not confirmed.", 422
    else:
        try:
            delete_conversation(conn, notebook_id, conversation_id)
        except ChatBusyError as e:
            error, status = str(e), 409
    if error is not None:
        if is_htmx(request):
            return _header(
                request,
                conn,
                notebook_id,
                conversation_id,
                "confirm",
                error=error,
                status_code=status,
            )
        return _page(
            request,
            conn,
            notebook_id,
            conversation_id,
            header_mode="confirm",
            header_error=error,
            status_code=status,
        )
    if is_htmx(request):
        return Response(status_code=204, headers={"HX-Redirect": chat_url(notebook_id)})
    return redirect(chat_url(notebook_id))
