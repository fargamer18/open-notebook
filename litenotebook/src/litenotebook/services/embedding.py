"""One lazily created embedder shared by ingestion and chat retrieval.

In production this is a ``WorkerEmbedder``: MiniLM runs in an isolated child
process that starts on the first embedding and exits after an idle period, so
torch/sentence-transformers never enter the web process. Tests inject a fake
via ``embedder_factory``. Rendering pages never creates or starts anything.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from litenotebook.config import Settings
    from litenotebook.embed.embedder import TextEmbedder

EmbedderFactory = Callable[[], "TextEmbedder"]


def worker_embedder_factory(settings: Settings | None = None) -> EmbedderFactory:
    def factory() -> TextEmbedder:
        from litenotebook.embed.worker_client import WorkerEmbedder

        if settings is None:
            return WorkerEmbedder()
        from litenotebook.packaging import embed_worker_command

        return WorkerEmbedder(
            idle_seconds=settings.embed_worker_idle_s,
            start_timeout=settings.embed_worker_start_timeout_s,
            command=embed_worker_command(settings),  # packaged: separate exe
        )

    return factory


def default_embedder_factory() -> TextEmbedder:
    return worker_embedder_factory()()


class SharedEmbedder:
    """A ``TextEmbedder`` that builds the real one on first use."""

    def __init__(self, factory: EmbedderFactory = default_embedder_factory) -> None:
        from litenotebook.config import EMBED_DIM, EMBED_MODEL

        self.model_name = EMBED_MODEL
        self.dim = EMBED_DIM
        self._factory = factory
        self._inner: TextEmbedder | None = None
        self._create_lock = threading.Lock()
        self._use_lock = threading.Lock()
        self.created = 0

    def _get(self) -> TextEmbedder:
        with self._create_lock:
            if self._inner is None:
                inner = self._factory()
                self.model_name, self.dim = inner.model_name, inner.dim
                self._inner = inner
                self.created += 1
            return self._inner

    def _serialized(self, inner: TextEmbedder) -> bool:
        # the worker client schedules requests itself (with priorities)
        return not getattr(inner, "thread_safe", False)

    @property
    def loaded(self) -> bool:
        inner = self._inner
        return inner is not None and bool(getattr(inner, "loaded", True))

    def load(self) -> None:
        inner = self._get()
        load = getattr(inner, "load", None)
        if callable(load):
            if self._serialized(inner):
                with self._use_lock:
                    load()
            else:
                load()

    def encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        inner = self._get()
        if self._serialized(inner):
            with self._use_lock:
                return inner.encode(texts)
        return inner.encode(texts)

    def status(self) -> dict[str, Any]:
        """Worker state for /healthz and Settings. Never starts anything."""
        inner = self._inner
        if inner is None:
            return {"status": "stopped"}
        status = getattr(inner, "status", None)
        if callable(status):
            result: dict[str, Any] = status()
            return result
        return {"status": "in-process"}

    def close(self) -> None:
        inner = self._inner
        close = getattr(inner, "close", None)
        if callable(close):
            close()
