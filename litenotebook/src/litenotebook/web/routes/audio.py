"""Study → Audio Overviews: create, generate, read, speak, play, download.

Only two actions use AI: "Generate" (MiniLM for a focus, Qwen for the
outline and script) and "Generate audio" (the local speech engine). The
history, transcript, player, chapters and downloads read SQLite and files
only — they work offline and start nothing.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response

from litenotebook.ai import speech
from litenotebook.ai.store import AIStore
from litenotebook.notebooks import get_notebook
from litenotebook.services.chat import QUIZ_BUSY_MESSAGE
from litenotebook.study import audio_store as store
from litenotebook.study.audio_model import (
    DEFAULT_WPM,
    DURATIONS,
    FORMAT_HELP,
    FORMAT_LABELS,
    FORMATS,
    MAX_FOCUS_CHARS,
    MAX_TITLE_CHARS,
    SPEAKER_LABELS,
    speakers_for,
)
from litenotebook.study.audio_render import safe_filename, transcript_text
from litenotebook.study.model import Material
from litenotebook.web.deps import (
    app_settings,
    get_db,
    is_htmx,
    llm_transport,
    redirect,
    render,
)
from litenotebook.web.routes.study import (
    CHAT_BUSY,
    NOT_CONFIGURED,
    _page,
    _ready_sources,
    study_url,
)

router = APIRouter(prefix="/notebooks/{notebook_id}/study/audio")
Db = Annotated[sqlite3.Connection, Depends(get_db)]
AUDIO_BUSY = "Local AI is busy writing an audio overview. Wait for it, or cancel it."
SPEECH_BUSY = "Speech is busy with another audio overview. Wait for it to finish."
NO_SPEECH = "Local speech is not installed."


def audio_url(notebook_id: int, rest: str = "") -> str:
    return study_url(notebook_id, f"/audio{rest}")


def _speech(request: Request) -> dict[str, Any]:
    return speech.status(app_settings(request))


def _busy_message(guard: Any) -> str:
    return {"quiz": QUIZ_BUSY_MESSAGE, "audio": AUDIO_BUSY}.get(guard.kind, CHAT_BUSY)


# --- history ----------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
def audio_history(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    get_notebook(conn, notebook_id)
    return _page(
        request,
        conn,
        notebook_id,
        "audio_list",
        overviews=store.list_overviews(conn, notebook_id),
        format_labels=FORMAT_LABELS,
    )


# --- create -----------------------------------------------------------------------


def _form_defaults(conn: sqlite3.Connection, notebook_id: int) -> dict[str, Any]:
    return {
        "sources": [s["id"] for s in _ready_sources(conn, notebook_id)],
        "notes": False,
        "chat": False,
        "focus": "",
        "title": "",
        "format": "two_person",
        "minutes": 10,
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
        "audio_new",
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
        formats=FORMATS,
        format_labels=FORMAT_LABELS,
        format_help=FORMAT_HELP,
        durations=DURATIONS,
        wpm=DEFAULT_WPM,
        max_focus=MAX_FOCUS_CHARS,
        max_title=MAX_TITLE_CHARS,
    )


@router.get("/new", response_class=HTMLResponse)
def new_overview(request: Request, conn: Db, notebook_id: int) -> HTMLResponse:
    get_notebook(conn, notebook_id)
    return _new_page(request, conn, notebook_id, _form_defaults(conn, notebook_id))


def default_title(names: list[str], focus: str) -> str:
    from litenotebook.web.routes.study import default_title as quiz_title

    return quiz_title(names, focus).removesuffix(" quiz") + " — audio overview"


@router.post("", response_model=None)
def create_overview(
    request: Request,
    conn: Db,
    notebook_id: int,
    sources: Annotated[list[str] | None, Form()] = None,
    notes: Annotated[str, Form()] = "",
    chat: Annotated[str, Form()] = "",
    focus: Annotated[str, Form()] = "",
    title: Annotated[str, Form()] = "",
    format: Annotated[str, Form()] = "two_person",  # noqa: A002 - form field name
    minutes: Annotated[str, Form()] = "10",
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
        "format": format,
        "minutes": int(minutes) if minutes.isdigit() else 10,
    }
    focus_clean, title_clean = " ".join(focus.split()), " ".join(title.split())
    error = None
    if any(sid not in ready for sid in chosen):
        error = "Choose sources from this notebook that are Ready."
    elif not chosen and not (form["notes"] or form["chat"]):
        error = "Select at least one source (or your notes or chat history)."
    elif format not in FORMATS:
        error = "Choose a format."
    elif form["minutes"] not in DURATIONS:
        error = "Choose a length of about 5, 10 or 20 minutes."
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
        return _new_page(
            request,
            conn,
            notebook_id,
            form,
            error=_busy_message(guard),
            status_code=409,
        )
    try:
        oid = _launch(request, conn, notebook_id, form, ready, focus_clean, title_clean)
    except BaseException:
        guard.release()
        raise
    return redirect(audio_url(notebook_id, f"/{oid}"))


def _launch(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    form: dict[str, Any],
    ready: dict[int, str],
    focus: str,
    title: str,
) -> int:
    from litenotebook.study.audio_generate import AudioJob, run_job
    from litenotebook.web.routes.chat import generation_target

    names = [ready[sid] for sid in form["sources"]]
    oid = store.create_overview(
        conn,
        notebook_id,
        title=title or default_title(names, focus),
        fmt=form["format"],
        minutes=form["minutes"],
        focus=focus,
        material={
            "sources": [{"id": sid, "filename": ready[sid]} for sid in form["sources"]],
            "notes": form["notes"],
            "chat": form["chat"],
        },
        wpm=DEFAULT_WPM,
    )
    guard = request.app.state.generation_guard
    job_owner = -(1_000_000_000 + oid)
    guard.bind(job_owner, "audio")
    llm, prepare, _setup = generation_target(request)
    job = AudioJob(
        settings=app_settings(request),
        notebook_id=notebook_id,
        overview_id=oid,
        material=Material(list(form["sources"]), form["notes"], form["chat"], focus),
        fmt=form["format"],
        minutes=form["minutes"],
        wpm=DEFAULT_WPM,
        llm=llm,
        guard=guard,
        prepare=prepare,
        embedder=request.app.state.embedder,
        managed=AIStore(conn).profile().launches_runtime,
        transport=llm_transport(request),
    )
    jobs: dict[int, Any] = request.app.state.audio_jobs
    jobs[oid] = job

    def work() -> None:
        try:
            run_job(job)
        finally:
            jobs.pop(oid, None)

    threading.Thread(target=work, name=f"audio-{oid}", daemon=True).start()
    return oid


@router.post("/{overview_id}/regenerate", response_model=None)
def regenerate(
    request: Request, conn: Db, notebook_id: int, overview_id: int
) -> Response:
    """A new overview with the same choices (the old one is never changed)."""
    o = store.get_overview(conn, notebook_id, overview_id)
    ready = {s["id"] for s in _ready_sources(conn, notebook_id)}
    kept = [s["id"] for s in o.material.get("sources", []) if s.get("id") in ready]
    return create_overview(
        request,
        conn,
        notebook_id,
        sources=[str(s) for s in kept],
        notes="1" if o.material.get("notes") else "",
        chat="1" if o.material.get("chat") else "",
        focus=o.focus,
        title="",
        format=o.format,
        minutes=str(o.target_minutes),
    )


# --- one overview -----------------------------------------------------------------


def _view(
    request: Request,
    conn: sqlite3.Connection,
    notebook_id: int,
    overview_id: int,
    *,
    status_code: int = 200,
    **extra: Any,
) -> HTMLResponse:
    o = store.get_overview(conn, notebook_id, overview_id)
    extra.setdefault("confirming", request.query_params.get("delete") == "1")
    secs = store.sections(conn, overview_id) if o.has_script else []
    ev = store.evidence(conn, overview_id) if o.has_script else {}
    return _page(
        request,
        conn,
        notebook_id,
        "audio_view",
        status_code=status_code,
        overview=o,
        audio_sections=secs,
        audio_evidence=ev,
        transcript_plain=transcript_text(o.title, secs, ev) if secs else "",
        assets=store.list_assets(conn, overview_id),
        speech=_speech(request),
        speakers=speakers_for(o.format),
        speaker_labels=SPEAKER_LABELS,
        format_labels=FORMAT_LABELS,
        **extra,
    )


@router.get("/{overview_id}", response_class=HTMLResponse)
def overview_page(
    request: Request, conn: Db, notebook_id: int, overview_id: int
) -> HTMLResponse:
    return _view(request, conn, notebook_id, overview_id)


@router.get("/{overview_id}/status", response_class=HTMLResponse)
def overview_status(
    request: Request, conn: Db, notebook_id: int, overview_id: int
) -> Response:
    o = store.get_overview(conn, notebook_id, overview_id)
    if not is_htmx(request):
        return redirect(audio_url(notebook_id, f"/{overview_id}"))
    response = render(
        request, "partials/audio_status.html", {"overview": o, "nb_id": notebook_id}
    )
    if not o.busy:
        response.headers["HX-Refresh"] = "true"
    return response


@router.post("/{overview_id}/cancel", response_model=None)
def cancel(request: Request, conn: Db, notebook_id: int, overview_id: int) -> Response:
    store.get_overview(conn, notebook_id, overview_id)
    for registry in (request.app.state.audio_jobs, request.app.state.speech_jobs):
        job = registry.get(overview_id)
        if job is not None:
            job.cancel.set()
    return redirect(audio_url(notebook_id, f"/{overview_id}"))


@router.post("/{overview_id}/synthesize", response_model=None)
def synthesize(
    request: Request,
    conn: Db,
    notebook_id: int,
    overview_id: int,
    voice_a: Annotated[str, Form()] = "",
    voice_b: Annotated[str, Form()] = "",
    voice_narrator: Annotated[str, Form()] = "",
    same_voice: Annotated[str, Form()] = "",
) -> Response:
    """Speak the (immutable) transcript into a new audio asset."""
    from litenotebook.study.audio_render import SynthJob, run_synthesis

    settings = app_settings(request)
    o = store.get_overview(conn, notebook_id, overview_id)
    if o.status not in ("script_ready", "ready"):
        return _view(
            request,
            conn,
            notebook_id,
            overview_id,
            status_code=409,
            synth_error="This overview has no finished transcript yet.",
        )
    provider = request.app.state.speech_provider(settings)
    if provider is None:
        return _view(
            request,
            conn,
            notebook_id,
            overview_id,
            status_code=409,
            synth_error=NO_SPEECH,
        )
    wanted = (
        {"A": voice_a, "B": voice_b}
        if speakers_for(o.format) == ("A", "B")
        else {"narrator": voice_narrator}
    )
    voices = {k: speech.voice(settings, v) for k, v in wanted.items()}
    if any(v is None for v in voices.values()):
        return _view(
            request,
            conn,
            notebook_id,
            overview_id,
            status_code=422,
            synth_error="Choose an installed voice for every speaker.",
        )
    if len(wanted) == 2 and voice_a == voice_b and same_voice != "1":
        return _view(
            request,
            conn,
            notebook_id,
            overview_id,
            status_code=422,
            synth_error="Both hosts would use the same voice. Install a second voice "
            "in Settings → Local AI, or tick “Use the same voice for both hosts”.",
        )
    lock: threading.Lock = request.app.state.speech_lock
    if not lock.acquire(blocking=False):
        return _view(
            request,
            conn,
            notebook_id,
            overview_id,
            status_code=409,
            synth_error=SPEECH_BUSY,
        )
    try:
        engine = speech.installed_engine(settings)
        engine_label = f"{provider.name} {engine.version}" if engine else provider.name
        asset_id = store.start_asset(conn, overview_id, engine_label, dict(wanted))
    except BaseException:
        lock.release()
        raise
    jobs: dict[int, Any] = request.app.state.speech_jobs

    def release() -> None:
        jobs.pop(overview_id, None)
        lock.release()

    job = SynthJob(
        settings=settings,
        notebook_id=notebook_id,
        overview_id=overview_id,
        asset_id=asset_id,
        voices={k: v for k, v in voices.items() if v is not None},
        provider=provider,
        release=release,
    )
    jobs[overview_id] = job
    threading.Thread(
        target=run_synthesis, args=(job,), name=f"speech-{overview_id}", daemon=True
    ).start()
    return redirect(audio_url(notebook_id, f"/{overview_id}"))


@router.get("/{overview_id}/audio/{asset_id}", response_model=None)
def audio_file(
    request: Request,
    conn: Db,
    notebook_id: int,
    overview_id: int,
    asset_id: int,
    download: int = 0,
) -> Response:
    """The WAV file (Range requests supported, so the player can seek)."""
    o = store.get_overview(conn, notebook_id, overview_id)
    asset = store.get_asset(conn, notebook_id, overview_id, asset_id)
    path = store.asset_path(app_settings(request), notebook_id, overview_id, asset_id)
    if asset.status != "ready" or not path.is_file():
        return PlainTextResponse("Audio not available", status_code=404)
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=safe_filename(o.title, ".wav") if download else None,
        content_disposition_type="attachment" if download else "inline",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{overview_id}/transcript.txt", response_model=None)
def transcript(
    request: Request, conn: Db, notebook_id: int, overview_id: int
) -> Response:
    o = store.get_overview(conn, notebook_id, overview_id)
    if not o.has_script:
        return PlainTextResponse("No transcript yet", status_code=404)
    text = transcript_text(
        o.title, store.sections(conn, overview_id), store.evidence(conn, overview_id)
    )
    name = safe_filename(o.title, ".txt")
    return PlainTextResponse(
        text,
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/{overview_id}/delete", response_class=HTMLResponse)
def delete_confirm(request: Request, notebook_id: int, overview_id: int) -> Response:
    return redirect(audio_url(notebook_id, f"/{overview_id}?delete=1"))


@router.post("/{overview_id}/delete", response_model=None)
def delete(
    request: Request,
    conn: Db,
    notebook_id: int,
    overview_id: int,
    confirm: Annotated[str, Form()] = "",
) -> Response:
    store.get_overview(conn, notebook_id, overview_id)
    error = None
    if confirm != str(overview_id):
        error, status = "Deletion not confirmed.", 422
    else:
        try:
            store.delete_overview(conn, app_settings(request), notebook_id, overview_id)
        except store.AudioError as e:
            error, status = str(e), 409
    if error is not None:
        return _view(
            request,
            conn,
            notebook_id,
            overview_id,
            status_code=status,
            confirming=True,
            delete_error=error,
        )
    if is_htmx(request):
        return Response(
            status_code=204, headers={"HX-Redirect": audio_url(notebook_id)}
        )
    return redirect(audio_url(notebook_id), flash="audio-deleted")
