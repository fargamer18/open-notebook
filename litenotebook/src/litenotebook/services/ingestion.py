"""In-process background ingestion: one worker thread, one job at a time.

Lifecycle inside the web app:

1. The upload request streams the PDF to disk, validates it and registers a
   ``pending`` source (``ingest.pipeline.register_source``), then calls
   ``enqueue`` and returns immediately.
2. The worker thread takes source ids from a FIFO queue and runs
   ``process_source`` with its own SQLite connection. Only one source is ever
   processed at a time, so there are no concurrent writes to a FAISS index and
   MiniLM runs in a single place.
3. MiniLM (and torch/faiss/pymupdf) are imported and loaded lazily by the first
   job, then kept for later jobs; rendering pages never loads them.
4. On startup the app fails any source a previous run left pending/processing
   (``sources.recover_interrupted``); nothing is re-run automatically. Retrying a
   failed source puts it back to ``pending`` and enqueues it again.
5. On shutdown the worker finishes nothing new: the queue is abandoned and a job
   in progress is left to be recovered as "interrupted" on the next start.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from litenotebook.config import Settings
from litenotebook.db import connect

if TYPE_CHECKING:
    from litenotebook.embed.embedder import TextEmbedder

log = logging.getLogger(__name__)

EmbedderFactory = Callable[[], "TextEmbedder"]


def default_embedder_factory() -> TextEmbedder:
    from litenotebook.embed.embedder import MiniLMEmbedder

    return MiniLMEmbedder()


class IngestionWorker:
    def __init__(
        self,
        settings: Settings,
        embedder_factory: EmbedderFactory = default_embedder_factory,
    ) -> None:
        self.settings = settings
        self.embedder_factory = embedder_factory
        self._queue: queue.Queue[int | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._jobs = 0  # enqueued but not yet finished
        self._jobs_cv = threading.Condition()
        self._embedder: TextEmbedder | None = None
        self.current: int | None = None

    # --- control ---------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="litenotebook-ingest", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout)
        self._thread = None

    def enqueue(self, source_id: int) -> None:
        with self._jobs_cv:
            self._jobs += 1
        self._queue.put(int(source_id))

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until every enqueued job has finished (tests and smoke checks)."""
        with self._jobs_cv:
            return self._jobs_cv.wait_for(lambda: self._jobs == 0, timeout)

    def _job_done(self) -> None:
        with self._jobs_cv:
            self._jobs -= 1
            self._jobs_cv.notify_all()

    @property
    def embedder_loaded(self) -> bool:
        return bool(getattr(self._embedder, "loaded", self._embedder is not None))

    # --- thread ----------------------------------------------------------------

    def _run(self) -> None:
        conn = connect(self.settings.db_path)
        try:
            while not self._stop.is_set():
                source_id = self._queue.get()
                if source_id is None:
                    break
                self.current = source_id
                try:
                    self._process(conn, source_id)
                except Exception:  # the worker must survive any single job
                    log.exception("Ingestion job for source %s crashed", source_id)
                finally:
                    self.current = None
                    self._job_done()
        finally:
            conn.close()

    def _process(self, conn: sqlite3.Connection, source_id: int) -> None:
        from litenotebook.index.faiss_store import FaissStore
        from litenotebook.ingest.pipeline import IngestError, process_source

        status = conn.execute(
            "SELECT status FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        if status is None or status[0] != "pending":
            return  # deleted, or already handled
        if self._embedder is None:
            self._embedder = self.embedder_factory()
        try:
            report = process_source(
                conn,
                self.settings,
                source_id,
                self._embedder,
                FaissStore(conn, self.settings),
            )
        except IngestError as e:
            log.info("Source %s failed: %s", source_id, e)
            return
        log.info(
            "Source %s ready: %d pages, %d chunks in %.1fs",
            source_id,
            report.pages,
            report.chunks,
            report.timings.get("total", 0.0),
        )
