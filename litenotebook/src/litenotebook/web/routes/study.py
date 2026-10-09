"""Study tab: quizzes — create, generate, take, review, retry and delete.

Only creating a quiz uses AI: it starts a background generation (MiniLM for
a focus, Qwen for the questions). Every other page — the Study home, quiz
history, taking a quiz, results and review — reads SQLite only, so it works
offline and never wakes MiniLM or llama-server.

Every route is scoped by the notebook id in the URL. Forms work without
JavaScript (POST → 303 redirect); HTMX only refreshes the generation status.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, Response

from litenotebook.ai.store import AIStore
from litenotebook.notebooks import get_notebook, get_summary
from litenotebook.services.chat import AUDIO_BUSY_MESSAGE, QUIZ_BUSY_MESSAGE
from litenotebook.study import store
from litenotebook.study.audio_store import list_overviews
from litenotebook.study.model import (
    COUNTS,
    DIFFICULTIES,
    DIFFICULTY_LABELS,
    KIND_LABELS,
    MAX_FOCUS_CHARS,
    MAX_TITLE_CHARS,
    QTYPE_LABELS,
    QTYPES,
    Material,
)
from litenotebook.web.deps import (
    app_settings,
    get_db,
    is_htmx,
    llm_transport,
    redirect,
    render,
)
from litenotebook.web.navigation import last_read

router = APIRouter(prefix="/notebooks/{notebook_id}/study")
Db = Annotated[sqlite3.Connection, Depends(get_db)]
NOT_CONFIGURED = "Local AI is not configured."
CHAT_BUSY = "Local AI is busy answering a chat question. Try again when it finishes."
CRUMBS = {
    "history": "All quizzes",
    "new": "New quiz",
    "take": "Questions",
    "result": "Result",
    "review": "Review",
    "audio_list": "Audio Overviews",
    "audio_new": "New audio overview",
}
STUDY_TOOLS = (  # Study sections (one card each on the Study home)
    ("quizzes", "Quizzes", "Create source-grounded quizzes from this notebook."),
    ("audio", "Audio Overviews", "Grounded podcast-style overviews of your sources."),
)


def study_url(notebook_id: int, rest: str = "") -> str:
    return f"/notebooks/{notebook_id}/study{rest}"


def _page(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    view: str,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    from litenotebook.web.routes.notebooks import WORKSPACE_TABS

    quiz = extra.get("quiz")
    overview = extra.get("overview")
    crumb = CRUMBS.get(view, "")
    if view == "quiz" and quiz is not None:
        crumb = quiz.title
    elif view == "audio_view" and overview is not None:
        crumb = overview.title
    return render(
        request,
        "notebook/workspace.html",
        {
            "s": get_summary(conn, notebook_id),
            "tabs": WORKSPACE_TABS,
            "tab": "study",
            "study_view": view,
            "study_crumb": crumb,
            "last_read": last_read(conn, notebook_id),
            "nb_id": notebook_id,
            "qtype_labels": QTYPE_LABELS,
            "difficulty_labels": DIFFICULTY_LABELS,
            "kind_labels": KIND_LABELS,
            **extra,
        },
        status_code=status_code,
    )


def _ready_sources(conn: sqlite3.Connection, notebook_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, original_filename, page_count FROM sources WHERE notebook_id = ? "
        "AND status = 'ready' ORDER BY id",
        (notebook_id,),
    ).fetchall()


def _jobs(request: Request) -> dict[int, Any]:
    jobs: dict[int, Any] = request.app.state.quiz_jobs
    return jobs


# --- Study home and history -------------------------------------------------------


@router.get("", response_class=HTMLResponse)
def study_home(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    get_notebook(conn, notebook_id)
    quizzes = store.list_quizzes(conn, notebook_id, limit=8)
    return _page(
        request,
        conn,
        notebook_id,
        "home",
        tools=STUDY_TOOLS,
        quizzes=quizzes,
        overviews=list_overviews(conn, notebook_id, limit=6),
        review=store.concept_progress(conn, notebook_id, needs_review=True, limit=12),
        progress=store.concept_progress(conn, notebook_id, limit=12),
        total_quizzes=conn.execute(
            "SELECT count(*) FROM quizzes WHERE notebook_id = ?", (notebook_id,)
        ).fetchone()[0],
    )


@router.get("/quizzes", response_class=HTMLResponse)
def quiz_history(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    get_notebook(conn, notebook_id)
    return _page(
        request,
        conn,
        notebook_id,
        "history",
        quizzes=store.list_quizzes(conn, notebook_id),
    )


# --- create -----------------------------------------------------------------------


def _form_defaults(conn: sqlite3.Connection, notebook_id: int) -> dict[str, Any]:
    sources = _ready_sources(conn, notebook_id)
    last = store.latest_settings(conn, notebook_id) or {}
    return {
        "sources": [s["id"] for s in sources],
        "notes": last.get("notes", False),
        "chat": last.get("chat", False),
        "focus": "",
        "title": "",
        "count": last.get("count", 10),
        "difficulty": last.get("difficulty", "medium"),
        "types": last.get("types", list(QTYPES)),
    }


def _new_page(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    form: dict[str, Any],
    *,
    error: str | None = None,
    setup_url: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    return _page(
        request,
        conn,
        notebook_id,
        "new",
        status_code=status_code,
        form=form,
        error=error,
        setup_url=setup_url,
        sources=_ready_sources(conn, notebook_id),
        notes_count=conn.execute(
            "SELECT count(*) FROM notes WHERE notebook_id = ?", (notebook_id,)
        ).fetchone()[0],
        chat_count=conn.execute(
            "SELECT count(*) FROM messages m JOIN conversations c ON c.id = "
            "m.conversation_id WHERE c.notebook_id = ? AND m.role = 'assistant' "
            "AND m.status = 'complete' AND m.model IS NOT NULL",
            (notebook_id,),
        ).fetchone()[0],
        counts=COUNTS,
        difficulties=DIFFICULTIES,
        qtypes=QTYPES,
        max_focus=MAX_FOCUS_CHARS,
        max_title=MAX_TITLE_CHARS,
    )


@router.get("/quizzes/new", response_class=HTMLResponse)
def new_quiz(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    get_notebook(conn, notebook_id)
    return _new_page(request, conn, notebook_id, _form_defaults(conn, notebook_id))


def default_title(names: list[str], focus: str) -> str:
    """No extra LLM call: the focus, else the selected source names."""
    if focus:
        base = focus
    else:
        stems = [
            n.rsplit(".", 1)[0] if n.lower().endswith(".pdf") else n for n in names
        ]
        if not stems:
            base = "Notes and chats"
        elif len(stems) == 1:
            base = stems[0]
        elif len(stems) == 2:
            base = f"{stems[0]} & {stems[1]}"
        else:
            base = f"{stems[0]} + {len(stems) - 1} more"
    base = " ".join(base.split())
    if len(base) > 80:
        base = base[:79].rstrip() + "…"
    return f"{base} quiz"


@router.post("/quizzes", response_model=None)
def create_quiz(
    request: Request,
    conn: Db,
    notebook_id: int,
    sources: Annotated[list[str] | None, Form()] = None,
    types: Annotated[list[str] | None, Form()] = None,
    notes: Annotated[str, Form()] = "",
    chat: Annotated[str, Form()] = "",
    focus: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "",
    count: Annotated[str, Form()] = "10",
    difficulty: Annotated[str, Form()] = "medium",
) -> Response:
    get_notebook(conn, notebook_id)
    ready = {s["id"]: s["original_filename"] for s in _ready_sources(conn, notebook_id)}
    chosen = [int(v) for v in sources or [] if v.isdigit()]
    form = {
        "sources": chosen,
        "notes": notes == "1",
        "chat": chat == "1",
        "focus": focus,
        "title": title,
        "count": int(count) if count.isdigit() else 10,
        "difficulty": difficulty,
        "types": [t for t in types or [] if t in QTYPES],
    }
    focus_clean = " ".join(focus.split())
    title_clean = " ".join(title.split())
    error = None
    if any(sid not in ready for sid in chosen):
        error = "Choose sources from this notebook that are Ready."
    elif not chosen and not (form["notes"] or form["chat"]):
        error = "Select at least one source (or your notes or chat history)."
    elif form["count"] not in COUNTS:
        error = "Choose 5, 10, 15 or 20 questions."
    elif difficulty not in DIFFICULTIES:
        error = "Choose a difficulty."
    elif not form["types"]:
        error = "Choose at least one question type."
    elif len(focus_clean) > MAX_FOCUS_CHARS:
        error = f"The focus can be at most {MAX_FOCUS_CHARS} characters."
    elif len(title_clean) > MAX_TITLE_CHARS:
        error = f"The title can be at most {MAX_TITLE_CHARS} characters."
    if error:
        return _new_page(request, conn, notebook_id, form, error=error, status_code=422)

    if AIStore(conn).profile().type == "none":
        return _new_page(
            request,
            conn,
            notebook_id,
            form,
            error=NOT_CONFIGURED,
            setup_url="/setup",
            status_code=409,
        )
    guard = request.app.state.generation_guard
    if not guard.try_acquire():
        busy = {"quiz": QUIZ_BUSY_MESSAGE, "audio": AUDIO_BUSY_MESSAGE}.get(
            guard.kind or "", CHAT_BUSY
        )
        return _new_page(request, conn, notebook_id, form, error=busy, status_code=409)
    try:
        quiz_id = _launch(
            request, conn, notebook_id, form, ready, focus_clean, title_clean
        )
    except BaseException:
        guard.release()
        raise
    return redirect(study_url(notebook_id, f"/quizzes/{quiz_id}"))


def _launch(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    form: dict[str, Any],
    ready: dict[int, str],
    focus: str,
    title: str,
) -> int:
    from litenotebook.study.generate import QuizJob, run_job
    from litenotebook.web.routes.chat import generation_target

    names = [ready[sid] for sid in form["sources"]]
    material = {
        "sources": [{"id": sid, "filename": ready[sid]} for sid in form["sources"]],
        "notes": form["notes"],
        "chat": form["chat"],
    }
    quiz_id = store.create_quiz(
        conn,
        notebook_id,
        title=title or default_title(names, focus),
        difficulty=form["difficulty"],
        types=form["types"],
        count=form["count"],
        focus=focus,
        material=material,
    )
    guard = request.app.state.generation_guard
    guard.bind(-quiz_id, "quiz")
    llm, prepare, _setup = generation_target(request)
    job = QuizJob(
        settings=app_settings(request),
        notebook_id=notebook_id,
        quiz_id=quiz_id,
        material=Material(
            source_ids=list(form["sources"]),
            include_notes=form["notes"],
            include_chat=form["chat"],
            focus=focus,
        ),
        difficulty=form["difficulty"],
        types=form["types"],
        count=form["count"],
        llm=llm,
        guard=guard,
        prepare=prepare,
        embedder=request.app.state.embedder,
        managed=AIStore(conn).profile().launches_runtime,
        transport=llm_transport(request),
    )
    jobs = _jobs(request)
    jobs[quiz_id] = job

    def work() -> None:
        try:
            run_job(job)
        finally:
            jobs.pop(quiz_id, None)

    threading.Thread(target=work, name=f"quiz-{quiz_id}", daemon=True).start()
    return quiz_id


# --- one quiz ---------------------------------------------------------------------


@router.get("/quizzes/{quiz_id}", response_class=HTMLResponse)
def quiz_page(
    request: Request, conn: Db, notebook_id: int, quiz_id: int
) -> HTMLResponse:
    quiz = store.get_quiz(conn, notebook_id, quiz_id)
    return _page(
        request,
        conn,
        notebook_id,
        "quiz",
        quiz=quiz,
        attempts=store.list_attempts(conn, quiz_id),
        confirming=request.query_params.get("delete") == "1",
    )


@router.get("/quizzes/{quiz_id}/status", response_class=HTMLResponse)
def quiz_status(request: Request, conn: Db, notebook_id: int, quiz_id: int) -> Response:
    quiz = store.get_quiz(conn, notebook_id, quiz_id)
    if not is_htmx(request):
        return redirect(study_url(notebook_id, f"/quizzes/{quiz_id}"))
    response = render(
        request,
        "partials/quiz_status.html",
        {"quiz": quiz, "nb_id": notebook_id},
    )
    if quiz.status != "generating":  # finished: reload the whole quiz page
        response.headers["HX-Refresh"] = "true"
    return response


@router.post("/quizzes/{quiz_id}/cancel", response_model=None)
def cancel_quiz(request: Request, conn: Db, notebook_id: int, quiz_id: int) -> Response:
    store.get_quiz(conn, notebook_id, quiz_id)
    job = _jobs(request).get(quiz_id)
    if job is not None:
        job.cancel.set()
    return redirect(study_url(notebook_id, f"/quizzes/{quiz_id}"))


@router.post("/quizzes/{quiz_id}/start", response_model=None)
def start(request: Request, conn: Db, notebook_id: int, quiz_id: int) -> Response:
    try:
        attempt_id = store.start_attempt(conn, notebook_id, quiz_id)
    except store.QuizNotFoundError:
        raise
    except store.StudyError:
        return redirect(study_url(notebook_id, f"/quizzes/{quiz_id}"))
    return redirect(study_url(notebook_id, f"/attempts/{attempt_id}"))


@router.get("/quizzes/{quiz_id}/delete", response_class=HTMLResponse)
def delete_confirm(
    request: Request, conn: Db, notebook_id: int, quiz_id: int
) -> Response:
    return redirect(study_url(notebook_id, f"/quizzes/{quiz_id}?delete=1"))


@router.post("/quizzes/{quiz_id}/delete", response_model=None)
def delete(
    request: Request,
    conn: Db,
    notebook_id: int,
    quiz_id: int,
    confirm: Annotated[str, Form()] = "",
) -> Response:
    quiz = store.get_quiz(conn, notebook_id, quiz_id)
    error = None
    if confirm != str(quiz_id):
        error = "Deletion not confirmed."
    else:
        try:
            store.delete_quiz(conn, notebook_id, quiz_id)
        except store.StudyError as e:
            error = str(e)
    if error is not None:
        return _page(
            request,
            conn,
            notebook_id,
            "quiz",
            quiz=quiz,
            attempts=store.list_attempts(conn, quiz_id),
            confirming=True,
            delete_error=error,
            status_code=409 if confirm == str(quiz_id) else 422,
        )
    if is_htmx(request):
        return Response(
            status_code=204, headers={"HX-Redirect": study_url(notebook_id)}
        )
    return redirect(study_url(notebook_id), flash="quiz-deleted")


# --- taking a quiz ----------------------------------------------------------------


def _attempt_view(
    conn: sqlite3.Connection, notebook_id: int, attempt_id: int
) -> tuple[store.Quiz, store.Attempt, dict[int, store.Question]]:
    quiz, attempt = store.get_attempt(conn, notebook_id, attempt_id)
    by_id = {q.id: q for q in store.questions(conn, quiz.id)}
    return quiz, attempt, by_id


@router.get("/attempts/{attempt_id}", response_class=HTMLResponse)
def take(
    request: Request,
    conn: Db,
    notebook_id: int,
    attempt_id: int,
    q: int | None = None,
    hint: int = 0,
) -> Response:
    quiz, attempt, by_id = _attempt_view(conn, notebook_id, attempt_id)
    if q is None:  # resume at the first unanswered question
        pending = [i for i in attempt.items if not i.answered]
        if attempt.status != "in_progress" or not pending:
            return redirect(study_url(notebook_id, f"/attempts/{attempt_id}/result"))
        q = pending[0].position
    item = next((i for i in attempt.items if i.position == q), None)
    if item is None:
        return redirect(study_url(notebook_id, f"/attempts/{attempt_id}"))
    return _page(
        request,
        conn,
        notebook_id,
        "take",
        quiz=quiz,
        attempt=attempt,
        item=item,
        question=by_id[item.question_id],
        show_hint=bool(hint) or item.hint_used,
        answer_error=request.query_params.get("error") == "1",
    )


@router.post("/attempts/{attempt_id}/answer", response_model=None)
def submit_answer(
    request: Request,
    conn: Db,
    notebook_id: int,
    attempt_id: int,
    q: Annotated[int, Form()],
    choice: Annotated[list[str] | None, Form()] = None,
) -> Response:
    selected = [int(c) for c in choice or [] if c.isdigit()]
    base = study_url(notebook_id, f"/attempts/{attempt_id}?q={q}")
    try:
        store.answer(conn, notebook_id, attempt_id, q, selected)
    except store.AttemptNotFoundError:
        raise
    except store.AlreadyAnsweredError:
        return redirect(base + "#feedback")
    except store.StudyError:
        return redirect(base + "&error=1#answer-error")
    return redirect(base + "#feedback")


@router.post("/attempts/{attempt_id}/hint", response_model=None)
def show_hint(
    request: Request,
    conn: Db,
    notebook_id: int,
    attempt_id: int,
    q: Annotated[int, Form()],
) -> Response:
    store.record_hint(conn, notebook_id, attempt_id, q)
    return redirect(study_url(notebook_id, f"/attempts/{attempt_id}?q={q}&hint=1#hint"))


@router.post("/attempts/{attempt_id}/finish", response_model=None)
def finish(request: Request, conn: Db, notebook_id: int, attempt_id: int) -> Response:
    store.finish_attempt(conn, notebook_id, attempt_id)
    return redirect(study_url(notebook_id, f"/attempts/{attempt_id}/result"))


@router.get("/attempts/{attempt_id}/result", response_class=HTMLResponse)
def result(request: Request, conn: Db, notebook_id: int, attempt_id: int) -> Response:
    quiz, attempt, by_id = _attempt_view(conn, notebook_id, attempt_id)
    if attempt.status == "in_progress":
        return redirect(study_url(notebook_id, f"/attempts/{attempt_id}"))
    return _page(
        request,
        conn,
        notebook_id,
        "result",
        quiz=quiz,
        attempt=attempt,
        by_id=by_id,
        to_review=store.attempt_concepts_to_review(conn, attempt, quiz.id),
    )


@router.get("/attempts/{attempt_id}/review", response_class=HTMLResponse)
def review(
    request: Request,
    conn: Db,
    notebook_id: int,
    attempt_id: int,
    only: str = "",
) -> HTMLResponse:
    quiz, attempt, by_id = _attempt_view(conn, notebook_id, attempt_id)
    items = attempt.missed if only == "mistakes" else attempt.items
    return _page(
        request,
        conn,
        notebook_id,
        "review",
        quiz=quiz,
        attempt=attempt,
        by_id=by_id,
        items=items,
        only=only,
    )


@router.post("/attempts/{attempt_id}/retry-mistakes", response_model=None)
def retry(request: Request, conn: Db, notebook_id: int, attempt_id: int) -> Response:
    try:
        new_id = store.retry_mistakes(conn, notebook_id, attempt_id)
    except store.AttemptNotFoundError:
        raise
    except store.StudyError:
        return redirect(study_url(notebook_id, f"/attempts/{attempt_id}/result"))
    return redirect(study_url(notebook_id, f"/attempts/{new_id}"))
