"""The chunker must never produce a chunk the model would truncate."""

import random

import pytest

from litenotebook.config import EMBED_MAX_TOKENS
from litenotebook.embed.embedder import count_tokens, get_tokenizer
from litenotebook.ingest.chunker import ChunkingError, chunk_page, chunk_pages

WORDS = (
    "matrix rank determinant eigenvalue Gaussian elimination pivot inconsistent "
    "homogeneous system solution vector subspace basis orthogonal projection"
).split()


def prose(seed: int, paragraphs: int = 10) -> str:
    rng = random.Random(seed)
    paras = []
    for _ in range(paragraphs):
        sentences = [
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 30))).capitalize()
            + rng.choice([".", "?", "!", ":"])
            for _ in range(rng.randint(1, 9))
        ]
        paras.append(" ".join(sentences))
    return "\n\n".join(paras)


ADVERSARIAL = {
    "prose": prose(1, 15),
    # No whitespace at all: only punctuation separates tokens (forces mid-run cuts).
    "no_whitespace": "ab,cd.ef;" * 600,
    # 80-char words split into many ## pieces. (Words over 100 chars become a
    # single [UNK] in WordPiece, so they can't overflow anything.)
    "long_single_words": " ".join("supercalifragilistic" * 4 for _ in range(200)),
    "digits": " ".join(str(10**30 + i) for i in range(300)),
    "math_unicode": " ".join(
        "∑ᵢ αᵢ·xᵢ ≤ ∞ ⇒ λ∈ℝ, A⁻¹ = adj(A)/det(A)." for _ in range(150)
    ),
    "cjk": "线性方程组的解。" * 400,
    "punctuation": "!?.,;:()[]{}" * 500,
    "single_lines": "\n".join(f"Line {i}: pivot row {i} swapped." for i in range(400)),
    "mixed_scripts": " ".join(
        "Ελλάδα Россия مرحبا नमस्ते café naïve" for _ in range(120)
    ),
}


def test_tokenizer_truncation_is_disabled():
    # tokenizer.json ships with truncation at 128; counting must see everything.
    assert count_tokens("word " * 1000) == 1002


def test_overlong_word_is_a_single_unknown_token():
    assert get_tokenizer().encode("x" * 500).tokens == ["[CLS]", "[UNK]", "[SEP]"]


def test_count_includes_special_tokens():
    assert get_tokenizer().encode("hello").tokens == ["[CLS]", "hello", "[SEP]"]
    assert count_tokens("hello") == 3


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_every_chunk_within_hard_limit(name):
    text = ADVERSARIAL[name]
    assert count_tokens(text) > EMBED_MAX_TOKENS  # would be truncated unchunked
    chunks = chunk_page(text, 1)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.n_tokens == count_tokens(chunk.text)  # true, re-tokenized count
        assert chunk.n_tokens <= EMBED_MAX_TOKENS
        assert text[chunk.char_start : chunk.char_end] == chunk.text


@pytest.mark.parametrize("seed", range(20))
def test_random_prose_stays_near_target_and_under_limit(seed):
    chunks = chunk_page(prose(seed, 25), 4, target_tokens=200, overlap_tokens=40)
    assert all(c.n_tokens <= EMBED_MAX_TOKENS for c in chunks)
    # all but the last chunk aim for ~200 content tokens (+2 specials)
    assert all(100 <= c.n_tokens <= 225 for c in chunks[:-1])


def test_chunks_cover_the_whole_page_in_order():
    text = ADVERSARIAL["prose"]
    chunks = chunk_page(text, 1)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text.rstrip())
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.char_start < nxt.char_start
        assert nxt.char_start <= prev.char_end  # no gaps


def test_consecutive_chunks_overlap_by_about_overlap_tokens():
    text = prose(7, 20)
    chunks = chunk_page(text, 1, target_tokens=200, overlap_tokens=40)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.char_start < prev.char_end
        shared = text[nxt.char_start : prev.char_end]
        assert 25 <= count_tokens(shared) - 2 <= 55


def test_zero_overlap():
    text = prose(3, 20)
    chunks = chunk_page(text, 1, overlap_tokens=0)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.char_start >= prev.char_end


def test_prefers_paragraph_and_sentence_breaks():
    text = prose(11, 30)
    chunks = chunk_page(text, 1)
    endings = [c.text[-1] for c in chunks[:-1]]
    assert sum(e in ".?!:" for e in endings) >= len(endings) * 0.8


def test_short_and_empty_pages():
    assert chunk_page("", 1) == []
    assert chunk_page("   \n\n  ", 1) == []
    [only] = chunk_page("A short page.", 5)
    assert (only.page_no, only.text, only.char_start) == (5, "A short page.", 0)


def test_chunks_never_cross_pages_and_keep_order():
    pages = [(1, prose(1, 8)), (2, ""), (3, "Tiny page three."), (4, prose(2, 8))]
    chunks = chunk_pages(pages)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    assert [c.page_no for c in chunks] == sorted(c.page_no for c in chunks)
    assert 2 not in {c.page_no for c in chunks}
    texts = dict(pages)
    for chunk in chunks:
        page_text = texts[chunk.page_no]
        assert page_text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.n_tokens <= EMBED_MAX_TOKENS


def test_invalid_parameters():
    with pytest.raises(ChunkingError):
        chunk_page("text", 1, target_tokens=255)  # 255 + 2 specials > 256
    with pytest.raises(ChunkingError):
        chunk_page("text", 1, target_tokens=100, overlap_tokens=100)
