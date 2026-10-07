"""Page-bounded chunking measured with the real MiniLM tokenizer.

Each page is chunked on its own, so a chunk never crosses a PDF page. Windows
of about ``target_tokens`` wordpieces advance with ``overlap_tokens`` of
overlap, and cut at the best nearby break (paragraph > line > sentence >
word). Every chunk's text is re-tokenized to prove it is within the model
limit — the embedder would otherwise silently truncate it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from litenotebook.config import EMBED_MAX_TOKENS
from litenotebook.embed.embedder import count_tokens, get_tokenizer

SPECIAL_TOKENS = 2  # [CLS] and [SEP]
_SENTENCE_END = (".", "?", "!", ":", ";")


class ChunkingError(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    page_no: int
    ord: int  # order within the source; assigned by chunk_pages()
    char_start: int  # offsets into the page text
    char_end: int
    text: str
    n_tokens: int  # including [CLS]/[SEP]
    heading: str | None = None


def chunk_page(
    text: str,
    page_no: int,
    *,
    target_tokens: int = 200,
    overlap_tokens: int = 40,
    max_tokens: int = EMBED_MAX_TOKENS,
    counter: Callable[[str], int] = count_tokens,
) -> list[Chunk]:
    """Split one page into overlapping chunks (``ord`` is left at 0)."""
    max_content = max_tokens - SPECIAL_TOKENS
    if not 0 < target_tokens <= max_content:
        raise ChunkingError(f"target_tokens must be in 1..{max_content}")
    if not 0 <= overlap_tokens < target_tokens:
        raise ChunkingError("overlap_tokens must be >= 0 and < target_tokens")

    encoding = get_tokenizer().encode(text, add_special_tokens=False)
    offsets = [(s, e) for s, e in encoding.offsets]
    pieces = encoding.tokens
    n = len(offsets)
    if n == 0:
        return []

    def score(k: int) -> int:
        """How good a cut is between token k-1 and token k."""
        gap = text[offsets[k - 1][1] : offsets[k][0]]
        if "\n\n" in gap:
            return 4
        if "\n" in gap:
            return 3
        if gap and pieces[k - 1] in _SENTENCE_END:
            return 2
        if gap:
            return 1
        return -1 if pieces[k].startswith("##") else 0

    def best_cut(lo: int, hi: int, ideal: int) -> int:
        """Best cut in [lo, hi]: highest score, then closest to ``ideal``."""
        return max(range(lo, hi + 1), key=lambda k: (score(k), -abs(k - ideal)))

    chunks: list[Chunk] = []
    start = 0
    while True:
        if n - start <= target_tokens:
            end = n
        else:
            ideal = start + target_tokens
            lo = start + max(1, target_tokens // 2)
            hi = min(ideal + target_tokens // 10, start + max_content, n - 1)
            end = best_cut(lo, hi, ideal)

        while True:
            char_start, char_end = offsets[start][0], offsets[end - 1][1]
            chunk_text = text[char_start:char_end]
            n_tokens = counter(chunk_text)
            if n_tokens <= max_tokens:
                break
            if end - start <= 1:
                raise ChunkingError(
                    f"Page {page_no}: a single token exceeds {max_tokens} tokens"
                )
            end -= 1  # re-tokenization disagreed with the slice; shrink

        chunks.append(
            Chunk(
                page_no=page_no,
                ord=0,
                char_start=char_start,
                char_end=char_end,
                text=chunk_text,
                n_tokens=n_tokens,
            )
        )
        if end >= n:
            return chunks

        # Step back by the overlap, landing on a word boundary near it.
        back = max(start + 1, end - overlap_tokens)
        window = range(max(start + 1, back - 8), min(end, back + 8) + 1)
        start = max(window, key=lambda k: (score(k) >= 1, -abs(k - back)))


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    *,
    target_tokens: int = 200,
    overlap_tokens: int = 40,
    max_tokens: int = EMBED_MAX_TOKENS,
) -> list[Chunk]:
    """Chunk ``(page_no, text)`` pairs in order and number chunks per source."""
    result: list[Chunk] = []
    for page_no, text in pages:
        for chunk in chunk_page(
            text,
            page_no,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
        ):
            result.append(
                Chunk(
                    page_no=chunk.page_no,
                    ord=len(result),
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    text=chunk.text,
                    n_tokens=chunk.n_tokens,
                    heading=chunk.heading,
                )
            )
    return result
