"""MiniLM tokenizer/embedder (CPU only) and embedding BLOB (de)serialization.

Qwen is never used here: embeddings always come from all-MiniLM-L6-v2 running
in-process on the CPU.

Local-first: the model and tokenizer are resolved from the local Hugging Face
cache with ``local_files_only=True`` and then loaded from that directory, so
loading never contacts huggingface.co (no metadata checks, no implicit
download), whatever the user's HF_HUB_OFFLINE setting. A missing model raises
``EmbeddingModelUnavailableError`` with instructions to download it once.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from litenotebook.config import (
    EMBED_DIM,
    EMBED_MAX_TOKENS,
    EMBED_MODEL,
    EMBED_REVISION,
)

if TYPE_CHECKING:
    from tokenizers import Tokenizer

FloatArray = npt.NDArray[np.float32]

BLOB_DTYPE = np.dtype("<f4")
BLOB_BYTES = EMBED_DIM * BLOB_DTYPE.itemsize
_NORM_TOLERANCE = 1e-3


# Files that must be in the local snapshot for an offline load.
REQUIRED_MODEL_FILES = ("modules.json", "config.json", "tokenizer.json")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


class EmbeddingError(ValueError):
    pass


class EmbeddingModelUnavailableError(EmbeddingError):
    """The embedding model is not in the local cache (it is never downloaded
    implicitly). The message is safe to show in the UI: it names no paths."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            f"The embedding model {EMBED_MODEL} is not available locally"
            f"{f' ({detail})' if detail else ''}. LiteNotebook does not download "
            "models during ingestion. Install it once from Settings → Local AI "
            "(Semantic search), or in a source checkout with "
            f"`hf download {EMBED_MODEL}`, then retry."
        )


class TextEmbedder(Protocol):
    model_name: str
    dim: int

    def encode(self, texts: list[str]) -> FloatArray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


@functools.lru_cache(maxsize=1)
def local_model_dir() -> Path:
    """The cached snapshot directory of the embedding model, found offline.

    ``snapshot_download(local_files_only=True)`` only reads the local cache
    (honouring HF_HOME / HF_HUB_CACHE); it never makes a network request.
    Failures are not cached, so downloading the model later just works.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = Path(
            snapshot_download(
                EMBED_MODEL, revision=EMBED_REVISION, local_files_only=True
            )
        )
    except (LocalEntryNotFoundError, OSError) as e:
        raise EmbeddingModelUnavailableError() from e
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).is_file()]
    if not any((path / name).is_file() for name in WEIGHT_FILES):
        missing.append("model weights")
    if missing:
        raise EmbeddingModelUnavailableError(f"missing {', '.join(missing)}")
    return path


@functools.lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    """The exact tokenizer all-MiniLM-L6-v2 uses, with truncation disabled.

    The model's tokenizer.json ships with truncation at 128 tokens; leaving
    it on would make every count silently cap out and hide oversized chunks.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(local_model_dir() / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def count_tokens(text: str) -> int:
    """Tokens the model sees for ``text``, including [CLS] and [SEP]."""
    return len(get_tokenizer().encode(text).ids)


class MiniLMEmbedder:
    """sentence-transformers all-MiniLM-L6-v2 on the CPU, loaded on first use."""

    model_name = EMBED_MODEL
    dim = EMBED_DIM

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self._model: object | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model now (otherwise it loads on the first ``encode``)."""
        self._load()

    def _load(self) -> object:
        if self._model is None:
            model_dir = local_model_dir()  # before the slow torch import
            from sentence_transformers import SentenceTransformer

            # A local directory plus local_files_only: no Hub requests at all.
            model = SentenceTransformer(
                str(model_dir), device="cpu", local_files_only=True
            )
            if model.max_seq_length != EMBED_MAX_TOKENS:
                raise EmbeddingError(
                    f"Unexpected max_seq_length {model.max_seq_length}"
                )
            self._model = model
        return self._model

    def encode(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        too_long = [
            i for i, t in enumerate(texts) if count_tokens(t) > EMBED_MAX_TOKENS
        ]
        if too_long:
            # Never rely on the model silently truncating input.
            raise EmbeddingError(
                f"{len(too_long)} text(s) exceed {EMBED_MAX_TOKENS} tokens "
                f"(first index {too_long[0]})"
            )
        model = self._load()
        vectors = model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = np.asarray(vectors, dtype=np.float32)
        validate_matrix(result)
        return result


def validate_vector(vector: npt.NDArray[np.floating], dim: int = EMBED_DIM) -> None:
    if vector.dtype != np.float32:
        raise EmbeddingError(f"Embedding dtype must be float32, got {vector.dtype}")
    if vector.shape != (dim,):
        raise EmbeddingError(f"Embedding shape must be ({dim},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise EmbeddingError("Embedding contains NaN or infinite values")
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > _NORM_TOLERANCE:
        raise EmbeddingError(f"Embedding is not L2-normalized (norm={norm:.4f})")


def validate_matrix(matrix: FloatArray, dim: int = EMBED_DIM) -> None:
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise EmbeddingError(f"Expected (n, {dim}) matrix, got {matrix.shape}")
    for row in matrix:
        validate_vector(row, dim)


def to_blob(vector: npt.NDArray[np.floating]) -> bytes:
    validate_vector(vector)
    blob = vector.astype(BLOB_DTYPE, copy=False).tobytes()
    if len(blob) != BLOB_BYTES:
        raise EmbeddingError(f"Embedding BLOB must be {BLOB_BYTES} bytes")
    return blob


def from_blob(blob: bytes) -> FloatArray:
    if len(blob) != BLOB_BYTES:
        raise EmbeddingError(
            f"Embedding BLOB must be {BLOB_BYTES} bytes, got {len(blob)}"
        )
    vector = np.frombuffer(blob, dtype=BLOB_DTYPE).astype(np.float32)
    validate_vector(vector)
    return vector
