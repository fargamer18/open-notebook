"""FastAPI application factory.

Rendering pages never loads MiniLM/torch/FAISS: the web layer imports only the
database, notebook, source and settings services. The ingestion worker imports
and loads the embedding stack on its first job.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from litenotebook import __version__
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
from litenotebook.web.deps import render
from litenotebook.web.routes import (
    chat,
    health,
    memory,
    notebooks,
    notes,
    search,
    settings,
    sources,
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
) -> FastAPI:
    """Build the app. ``env``, ``llm_transport`` and ``embedder_factory`` are
    injection points for tests. The ingestion worker runs only inside the
    lifespan (i.e. under uvicorn or ``with TestClient(app)``)."""
    app_settings = app_settings or load_settings(env)
    app_settings.data_dir.mkdir(parents=True, exist_ok=True)
    open_db(app_settings.db_path).close()  # apply migrations once at startup

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Requests use short-lived connections. If none stays open, SQLite
        # checkpoints and deletes the WAL file every time the last one closes,
        # and recreating it costs 0.1-1 s per write on slow filesystems such as
        # NTFS via FUSE. This idle connection keeps the WAL file in place.
        keeper = connect(app_settings.db_path)
        interrupted = recover_interrupted(keeper)
        if interrupted:
            log.warning("Interrupted sources marked failed: %s", interrupted)
        clean_stale_uploads(app_settings)
        if interrupted_answers := recover_generating(keeper):
            log.warning("Interrupted answers marked: %d", interrupted_answers)
        # Deterministic, idempotent: derive graph memory for any exchange that
        # has none yet (e.g. conversations from before Phase 5). No LLM calls.
        if remembered := backfill(keeper):
            log.info("Memory graph backfilled for %d exchange(s)", remembered)
        worker.start()
        try:
            yield
        finally:
            worker.stop()
            embedder.close()  # stops and reaps the MiniLM child process
            keeper.close()

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

    app.add_middleware(LocalOnlyMiddleware)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(notebooks.router)
    app.include_router(sources.router)
    app.include_router(chat.router)
    app.include_router(memory.router)
    app.include_router(notes.router)
    app.include_router(search.router)
    app.include_router(settings.router)
    app.include_router(health.router)

    @app.exception_handler(NotebookNotFoundError)
    async def not_found(request: Request, exc: NotebookNotFoundError) -> HTMLResponse:
        return render(
            request,
            "error.html",
            {"title": "Not found", "message": str(exc)},
            status_code=404,
        )

    @app.exception_handler(SourceNotFoundError)
    async def source_not_found(
        request: Request, exc: SourceNotFoundError
    ) -> HTMLResponse:
        return render(
            request,
            "error.html",
            {"title": "Not found", "message": str(exc)},
            status_code=404,
        )

    async def chat_not_found(request: Request, exc: Exception) -> HTMLResponse:
        return render(
            request,
            "error.html",
            {"title": "Not found", "message": str(exc)},
            status_code=404,
        )

    for missing in (
        ConversationNotFoundError,
        MessageNotFoundError,
        CitationNotFoundError,
        MemoryRefNotFoundError,
        memory.MemoryNodeNotFoundError,
        NoteNotFoundError,
    ):
        app.add_exception_handler(missing, chat_not_found)

    @app.exception_handler(SettingsError)
    async def bad_settings(request: Request, exc: SettingsError) -> HTMLResponse:
        return render(
            request,
            "error.html",
            {"title": "Invalid configuration", "message": str(exc)},
            status_code=500,
        )

    return app
