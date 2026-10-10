"""Page text geometry and deterministic chunk → page-region matching.

Words come from PyMuPDF with their boxes mapped through the page rotation and
normalised to 0..1 of the rendered page. Matching a stored chunk works on a
normalised character stream (letters/digits only, case-folded), so line breaks,
whitespace, punctuation and hyphenation across lines do not matter. When the
same passage occurs more than once on a page, the occurrence nearest to the
chunk's stored offset wins. If nothing matches, no rectangles are returned —
an unrelated region is never highlighted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
EDGE_CHARS = 60  # anchor length for the prefix/suffix fallback


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block: int
    line: int

    def as_dict(self) -> dict[str, object]:
        return {
            "t": self.text,
            "b": [
                round(self.x0, 5),
                round(self.y0, 5),
                round(self.x1, 5),
                round(self.y1, 5),
            ],
            "l": f"{self.block}.{self.line}",
        }


@dataclass(frozen=True)
class PageGeometry:
    page_no: int
    width: float
    height: float
    words: list[Word]

    @property
    def has_text(self) -> bool:
        return bool(self.words)


def page_geometry(path: Path, page_no: int) -> PageGeometry:
    import pymupdf

    from litenotebook.ingest.pdflock import PDF_LOCK

    with PDF_LOCK, pymupdf.open(path, filetype="pdf") as doc:
        page = doc[page_no - 1]
        rect = page.rect  # rotation applied
        width, height = rect.width or 1.0, rect.height or 1.0
        matrix = page.rotation_matrix
        words = []
        for x0, y0, x1, y1, text, block, line, _ in page.get_text("words", sort=True):
            box = pymupdf.Rect(x0, y0, x1, y1) * matrix
            box.normalize()
            nx0 = min(max(box.x0 / width, 0.0), 1.0)
            ny0 = min(max(box.y0 / height, 0.0), 1.0)
            nx1 = min(max(box.x1 / width, 0.0), 1.0)
            ny1 = min(max(box.y1 / height, 0.0), 1.0)
            if nx1 <= nx0 or ny1 <= ny0 or not text.strip():
                continue
            words.append(Word(text, nx0, ny0, nx1, ny1, int(block), int(line)))
    return PageGeometry(page_no, float(width), float(height), words)


def _norm(text: str) -> str:
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", text).casefold())


def _stream(words: list[Word]) -> tuple[str, list[int]]:
    """Normalised page text and, per character, the index of its word."""
    chars: list[str] = []
    owner: list[int] = []
    for i, word in enumerate(words):
        norm = _norm(word.text)
        chars.append(norm)
        owner.extend([i] * len(norm))
    return "".join(chars), owner


def _occurrences(haystack: str, needle: str) -> list[int]:
    found, start = [], haystack.find(needle)
    while start >= 0:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return found


def match_text(
    geometry: PageGeometry, text: str, position_hint: float | None = None
) -> list[list[float]] | None:
    """Rectangles (normalised, one per text line) covering ``text`` on the
    page, or None if it cannot be located with confidence.

    ``position_hint`` (0..1) is where the passage starts within the page text;
    it picks between repeated occurrences.
    """
    page, owner = _stream(geometry.words)
    needle = _norm(text)
    if not needle or not page:
        return None
    span: tuple[int, int] | None = None
    starts = _occurrences(page, needle)
    if starts:
        start = _closest(starts, len(page), position_hint)
        span = (start, start + len(needle))
    elif len(needle) > 2 * EDGE_CHARS:
        heads = _occurrences(page, needle[:EDGE_CHARS])
        if heads:
            head = _closest(heads, len(page), position_hint)
            tail = page.find(needle[-EDGE_CHARS:], head + EDGE_CHARS)
            if tail >= 0:
                end = tail + EDGE_CHARS
                if end - head <= 1.25 * len(needle):  # plausibly the same passage
                    span = (head, end)
    if span is None:
        return None
    first, last = owner[span[0]], owner[span[1] - 1]
    return _line_rects(geometry.words[first : last + 1])


def _closest(starts: list[int], total: int, hint: float | None) -> int:
    if hint is None or len(starts) == 1:
        return starts[0]
    target = hint * total
    return min(starts, key=lambda s: (abs(s - target), s))


def _line_rects(words: list[Word]) -> list[list[float]]:
    lines: dict[tuple[int, int], list[float]] = {}
    for w in words:
        key = (w.block, w.line)
        if key not in lines:
            lines[key] = [w.x0, w.y0, w.x1, w.y1]
        else:
            box = lines[key]
            box[0], box[1] = min(box[0], w.x0), min(box[1], w.y0)
            box[2], box[3] = max(box[2], w.x1), max(box[3], w.y1)
    return [[round(v, 5) for v in box] for box in lines.values()]


def text_offset_hint(page_text: str, char_start: int) -> float | None:
    """Where a chunk starts within its stored page text (0..1)."""
    if not page_text:
        return None
    return max(0.0, min(1.0, char_start / len(page_text)))
