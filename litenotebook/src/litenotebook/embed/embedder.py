"""MiniLM tokenizer/embedder (CPU only) and embedding BLOB (de)serialization.

Qwen is never used here: embeddings always come from all-MiniLM-L6-v2 running
in-process on the CPU.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from litenotebook.config import EMBED_DIM, EMBED_MAX_TOKENS, EMBED_MODEL

if TYPE_CHECKING:
    from tokenizers import Tokenizer

FloatArray = npt.NDArray[np.float32]

BLOB_DTYPE = np.dtype("<f4")
BLOB_BYTES = EMBED_DIM * BLOB_DTYPE.itemsize
_NORM_TOLERANCE = 1e-3


class EmbeddingError(ValueError):
    pass


class TextEmbedder(Protocol):
    model_name: str
    dim: int

    def encode(self, texts: list[str]) -> FloatArray:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


@functools.lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    """The exact tokenizer all-MiniLM-L6-v2 uses, with truncation disabled.

    The model's tokenizer.json ships with truncation at 128 tokens; leaving
    it on would make every count silently cap out and hide oversized chunks.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(EMBED_MODEL, "tokenizer.json"))
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

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(EMBED_MODEL, device="cpu")
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
