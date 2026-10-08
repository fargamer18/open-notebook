"""One lazily created MiniLM embedder shared by ingestion and chat retrieval.

The model is created on first use only (never while rendering pages) and
then reused, so the ~15 s cold load and ~500 MB happen at most once per
process. ``encode`` calls are serialized: the ingestion worker thread and a
chat question may embed at the same time.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from litenotebook.embed.embedder import TextEmbedder

EmbedderFactory = Callable[[], "TextEmbedder"]


def default_embedder_factory() -> TextEmbedder:
    from litenotebook.embed.embedder import MiniLMEmbedder

    return MiniLMEmbedder()


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

    @property
    def loaded(self) -> bool:
        inner = self._inner
        return inner is not None and bool(getattr(inner, "loaded", True))

    def load(self) -> None:
        inner = self._get()
        load = getattr(inner, "load", None)
        if callable(load):
            with self._use_lock:
                load()

    def encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        inner = self._get()
        with self._use_lock:
            return inner.encode(texts)
