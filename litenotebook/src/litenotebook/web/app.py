"""FastAPI application factory.

Rendering pages never loads MiniLM/torch/FAISS: the web layer imports only the
database, notebook, source and settings services. The ingestion worker imports
and loads the embedding stack on its first job.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from litenotebook import __version__
from litenotebook.ai.jobs import JobManager
from litenotebook.ai.runtime_manager import RuntimeManager
from litenotebook.ai.service import LocalAI
from litenotebook.ai.speech import SpeechProvider, provider_for
from litenotebook.chat import (
    CitationNotFoundError,
    ConversationNotFoundError,
    MemoryRefNotFoundError,
    MessageNotFoundError,
    recover_generating,
)
from litenotebook.config import Settings, load_settings
from litenotebook.db import connect, open_db
from litenotebook.memory.graph import backfill
from litenotebook.notebooks import NotebookNotFoundError
from litenotebook.notes import NoteNotFoundError
from litenotebook.platform_utils.filesystem import InstanceLock, remove_tree
from litenotebook.reader.annotations import AnnotationNotFoundError
from litenotebook.services.chat import GenerationGuard
from litenotebook.services.embedding import (
    EmbedderFactory,
    SharedEmbedder,
    worker_embedder_factory,
)
from litenotebook.services.ingestion import IngestionWorker
from litenotebook.services.settings import SettingsError
from litenotebook.services.uploads import clean_stale_uploads
from litenotebook.sources import SourceNotFoundError, recover_interrupted
from litenotebook.study.audio_store import AssetNotFoundError, OverviewNotFoundError
from litenotebook.study.audio_store import recover as recover_audio
from litenotebook.study.store import AttemptNotFoundError, QuizNotFoundError
from litenotebook.study.store import recover_generating as recover_quizzes
from litenotebook.web.deps import render
from litenotebook.web.routes import (
    ai,
    audio,
    backup,
    chat,
    health,
    memory,
    notebooks,
    notes,
    reader,
    search,
    settings,
    shell,
    sources,
    study,
)
from litenotebook.web.security import LocalOnlyMiddleware

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger(__name__)


def create_app(
    app_settings: Settings | None = None,
    *,
    env: Mapping[str, str] | None = None,
    llm_transport: httpx.BaseTransport | None = None,
    embedder_factory: EmbedderFactory | None = None,
    local_ai: LocalAI | None = None,
    speech_provider: Callable[[Settings], SpeechProvider | None] | None = None,
) -> FastAPI:
    """Build the app. ``env``, ``llm_transport`` and ``embedder_factory`` are
    injection points for tests. The ingestion worker runs only inside the
    lifespan (i.e. under uvicorn or ``with TestClient(app)``)."""
    app_settings = app_settings or load_settings(env)
    _apply_staged_restore(app_settings)  # before anything opens the database
    app_settings.data_dir.mkdir(parents=True, exist_ok=True)
    open_db(app_settings.db_path).close()  # apply migrations once at startup

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Requests use short-lived connections. If none stays open, SQLite
        # checkpoints and deletes the WAL file every time the last one closes,
        # and recreating it costs 0.1-1 s per write on slow filesystems such as
        # NTFS via FUSE. This idle connection keeps the WAL file in place.
        # Advisory "a server uses this data directory" lock: a restore waits
        # for it to be released instead of swapping files under a live server.
        # One writer per data folder: refuse to start (clear message) if
        # another LiteNotebook process holds it. The OS drops the lock on exit
        # or crash, so a left-over lock file never blocks a later start.
        instance = InstanceLock(app_settings.data_dir).acquire_or_raise("server")
        keeper = connect(app_settings.db_path)
        interrupted = recover_interrupted(keeper)
        if interrupted:
            log.warning("Interrupted sources marked failed: %s", interrupted)
        clean_stale_uploads(app_settings)
        clean_archive_tmp(app_settings)
        if interrupted_answers := recover_generating(keeper):
            log.warning("Interrupted answers marked: %d", interrupted_answers)
        if interrupted_quizzes := recover_quizzes(keeper):
            log.warning("Interrupted quiz generations marked: %d", interrupted_quizzes)
        scripts, speech_runs = recover_audio(keeper, app_settings)
        if scripts or speech_runs:
            log.warning(
                "Interrupted audio overviews: %d script(s), %d synthesis run(s)",
                scripts,
                speech_runs,
            )
        # Deterministic, idempotent: derive graph memory for any exchange that
        # has none yet (e.g. conversations from before Phase 5). No LLM calls.
        if remembered := backfill(keeper):
            log.info("Memory graph backfilled for %d exchange(s)", remembered)
        worker.start()
        try:
            yield
        finally:
            worker.stop()
            for registry in (
                app.state.quiz_jobs,
                app.state.audio_jobs,
                app.state.speech_jobs,
            ):
                for job in list(registry.values()):
                    job.cancel.set()  # threads stop at their next check
            local_ai_service.runtime.stop()  # only the llama-server this app started
            embedder.close()  # stops and reaps the MiniLM child process
            keeper.close()
            instance.release()

    # One lazily created MiniLM shared by ingestion and chat retrieval.
    embedder = SharedEmbedder(embedder_factory or worker_embedder_factory(app_settings))
    worker = IngestionWorker(app_settings, lambda: embedder)

    app = FastAPI(
        title="LiteNotebook",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.env = env
    app.state.llm_transport = llm_transport
    app.state.worker = worker
    app.state.embedder = embedder
    app.state.generation_guard = GenerationGuard()
    # Nothing starts here: llama-server starts on request (or autostart when
    # Chat needs it); hardware is scanned only when a setup page asks.
    local_ai_service = local_ai or LocalAI(app_settings, RuntimeManager(app_settings))
    app.state.local_ai = local_ai_service
    app.state.ai_jobs = JobManager()
    app.state.quiz_jobs = {}  # quiz id → running QuizJob (for Cancel)
    app.state.audio_jobs = {}  # overview id → running script generation
    app.state.speech_jobs = {}  # overview id → running synthesis
    app.state.speech_lock = threading.Lock()  # one speech job at a time
    app.state.speech_provider = speech_provider or provider_for

    app.add_middleware(LocalOnlyMiddleware)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(notebooks.router)
    app.include_router(sources.router)
    app.include_router(chat.router)
    app.include_router(memory.router)
    app.include_router(notes.router)
    app.include_router(search.router)
    app.include_router(reader.router)
    app.include_router(reader.panels)
    app.include_router(settings.router)
    app.include_router(health.router)
    app.include_router(shell.router)
    app.include_router(backup.router)
    app.include_router(ai.router)
    app.include_router(study.router)
    app.include_router(audio.router)

    async def missing_item(request: Request, exc: Exception) -> HTMLResponse:
        # A missing notebook gets no "back to notebook" link.
        in_notebook = not isinstance(exc, NotebookNotFoundError)
        return error_page(request, 404, str(exc), in_notebook=in_notebook)

    for missing in (
        NotebookNotFoundError,
        SourceNotFoundError,
        ConversationNotFoundError,
        MessageNotFoundError,
        CitationNotFoundError,
        MemoryRefNotFoundError,
        memory.MemoryNodeNotFoundError,
        NoteNotFoundError,
        AnnotationNotFoundError,
        QuizNotFoundError,
        AttemptNotFoundError,
        OverviewNotFoundError,
        AssetNotFoundError,
    ):
        app.add_exception_handler(missing, missing_item)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        if not wants_page(request):  # API/HTMX callers keep FastAPI's JSON
            return await http_exception_handler(request, exc)
        return error_page(request, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        request: Request, exc: RequestValidationError
    ) -> Response:
        if not wants_page(request):
            return await request_validation_exception_handler(request, exc)
        return error_page(request, 422)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> Response:
        # Details go to the local server log only, never to the page.
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        if not wants_page(request):
            return JSONResponse({"error": "Unexpected error"}, status_code=500)
        return error_page(request, 500)

    @app.exception_handler(SettingsError)
    async def bad_settings(request: Request, exc: SettingsError) -> HTMLResponse:
        return render(
            request,
            "error.html",
            {"title": "Invalid configuration", "message": str(exc)},
            status_code=500,
        )

    return app


def _apply_staged_restore(settings: Settings) -> None:
    from litenotebook.archive.restore import RestoreBusyError, apply_pending_restore

    try:
        applied = apply_pending_restore(settings.data_dir)
    except RestoreBusyError as e:
        log.warning("%s", e)
        return
    if applied is not None:
        log.warning(
            "Restore applied from backup created %s; previous data kept as %s",
            applied.marker.get("archive_created_at"),
            applied.safety_dir.name if applied.safety_dir else "(none)",
        )


def clean_archive_tmp(settings: Settings) -> int:
    """Archive uploads, unsent downloads and import staging left by an
    interrupted run (startup only, so nothing is in use)."""
    root = settings.archive_tmp_dir
    if not root.is_dir() or root.is_symlink():
        return 0
    removed = 0
    for path in root.iterdir():
        with contextlib.suppress(OSError):
            remove_tree(path)
            removed += 1
    return removed


ERROR_PAGES = {
    403: ("Not allowed", "This request is not allowed."),
    404: ("Not found", "That page or item does not exist (it may have been deleted)."),
    405: ("Not allowed", "That action is not available here."),
    422: ("Invalid request", "The address or form contained an invalid value."),
    500: (
        "Something went wrong",
        "An unexpected error occurred. Details are in the local server log.",
    ),
}


def wants_page(request: Request) -> bool:
    """A browser navigation (not HTMX, not fetch/JSON)."""
    return request.headers.get(
        "hx-request"
    ) != "true" and "text/html" in request.headers.get("accept", "")


def error_page(
    request: Request, status: int, message: str = "", *, in_notebook: bool = True
) -> HTMLResponse:
    title, default = ERROR_PAGES.get(status, ("Error", "The request failed."))
    raw = request.path_params.get("notebook_id") if in_notebook else None
    notebook_id = int(raw) if isinstance(raw, str) and raw.isdigit() else None
    return render(
        request,
        "error.html",
        {
            "title": title,
            "message": message or default,
            "status": status,
            "notebook_id": notebook_id,
        },
        status_code=status,
    )
