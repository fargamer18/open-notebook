import numpy as np
import pytest

from litenotebook.config import EMBED_DIM
from litenotebook.embed.embedder import (
    BLOB_BYTES,
    EmbeddingError,
    MiniLMEmbedder,
    count_tokens,
    from_blob,
    to_blob,
    validate_matrix,
)


def unit(dim: int = EMBED_DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_blob_roundtrip_is_exact():
    v = unit()
    blob = to_blob(v)
    assert isinstance(blob, bytes) and len(blob) == BLOB_BYTES == 384 * 4
    out = from_blob(blob)
    assert out.dtype == np.float32 and out.shape == (384,)
    assert np.array_equal(out, v)


def test_blob_is_little_endian_float32():
    v = unit()
    assert to_blob(v) == v.astype("<f4").tobytes()


@pytest.mark.parametrize(
    "vector",
    [
        unit(dim=383),
        unit(dim=768),
        unit().astype(np.float64),
        unit() * 2,  # not normalized
        np.full(384, np.nan, dtype=np.float32),
        np.concatenate([[np.inf], np.zeros(383)]).astype(np.float32),
    ],
    ids=["dim383", "dim768", "float64", "unnormalized", "nan", "inf"],
)
def test_to_blob_rejects_invalid_vectors(vector):
    with pytest.raises(EmbeddingError):
        to_blob(vector)


@pytest.mark.parametrize("size", [0, 4, BLOB_BYTES - 4, BLOB_BYTES + 4, 768 * 4])
def test_from_blob_rejects_wrong_length(size):
    with pytest.raises(EmbeddingError):
        from_blob(b"\x00" * size)


def test_from_blob_rejects_non_finite():
    v = unit()
    v[3] = np.nan
    with pytest.raises(EmbeddingError):
        from_blob(v.astype("<f4").tobytes())


def test_validate_matrix_shape():
    with pytest.raises(EmbeddingError):
        validate_matrix(np.zeros((2, 100), dtype=np.float32))


def test_embedder_refuses_over_long_text_before_loading_model():
    embedder = MiniLMEmbedder()
    with pytest.raises(EmbeddingError, match="exceed 256"):
        embedder.encode(["word " * 300])
    assert embedder._model is None  # never relied on silent truncation


def test_empty_batch():
    assert MiniLMEmbedder().encode([]).shape == (0, EMBED_DIM)


@pytest.mark.model
def test_real_minilm_on_cpu():
    embedder = MiniLMEmbedder()
    vectors = embedder.encode(
        ["Gaussian elimination solves linear systems.", "Cats purr."]
    )
    validate_matrix(vectors)
    assert str(embedder._model.device) == "cpu"  # type: ignore[union-attr]
    related = embedder.encode(["How do you solve a system of linear equations?"])[0]
    assert related @ vectors[0] > related @ vectors[1]


@pytest.mark.model
def test_our_token_counts_match_sentence_transformers():
    embedder = MiniLMEmbedder()
    model = embedder._load()
    for text in ["hello", "Gaussian elimination!", "∑ αᵢ ≤ ∞", "word " * 400]:
        ids = model.tokenizer(text, truncation=False)["input_ids"]  # type: ignore[attr-defined]
        assert len(ids) == count_tokens(text)
