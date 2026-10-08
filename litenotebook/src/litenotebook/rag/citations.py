"""Parse ``[S#]`` citations and resolve them against the supplied context.

Source/page/chunk metadata always comes from the retrieval results, never from
the model's text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from litenotebook.retrieval.hybrid import SearchResult

# [S1]  [S1, S2]  [S1,S2; S3]  [S1, 2]  [M1]  [M1][M2]  [S1, M2]
# (S = source evidence, M = memory; a bare number repeats the previous kind)
REF_GROUP = re.compile(r"\[\s*[SM]\s*\d+(?:\s*[,;]\s*[SM]?\s*\d+)*\s*\]", re.IGNORECASE)
_ITEM = re.compile(r"([SM]?)\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Citation:
    label: int
    chunk_id: int
    source_id: int
    source_filename: str
    page_no: int
    ord: int


@dataclass(frozen=True)
class CitationReport:
    citations: list[Citation]  # valid, in order of first appearance
    invalid_labels: list[int]  # cited but not in the supplied context


@dataclass(frozen=True)
class RefSpan:
    start: int
    end: int
    refs: list[tuple[str, int]]  # ("S" | "M", label) in written order


def ref_spans(text: str) -> list[RefSpan]:
    spans: list[RefSpan] = []
    for match in REF_GROUP.finditer(text):
        refs: list[tuple[str, int]] = []
        kind = "S"
        for prefix, number in _ITEM.findall(match.group(0)):
            kind = prefix.upper() or kind
            refs.append((kind, int(number)))
        spans.append(RefSpan(match.start(), match.end(), refs))
    return spans


def parse_refs(text: str) -> list[tuple[str, int]]:
    """All cited (kind, label) pairs in order of first appearance."""
    seen: list[tuple[str, int]] = []
    for span in ref_spans(text):
        for ref in span.refs:
            if ref not in seen:
                seen.append(ref)
    return seen


def parse_labels(text: str) -> list[int]:
    """Cited source labels ([S#]) in order of first appearance."""
    return [label for kind, label in parse_refs(text) if kind == "S"]


def parse_memory_labels(text: str) -> list[int]:
    """Cited memory labels ([M#]) in order of first appearance."""
    return [label for kind, label in parse_refs(text) if kind == "M"]


def resolve(text: str, results: Sequence[SearchResult]) -> CitationReport:
    citations: list[Citation] = []
    invalid: list[int] = []
    for label in parse_labels(text):
        if 1 <= label <= len(results):
            r = results[label - 1]
            citations.append(
                Citation(
                    label, r.chunk_id, r.source_id, r.source_filename, r.page_no, r.ord
                )
            )
        else:
            invalid.append(label)
    return CitationReport(citations, invalid)


def format_citation(c: Citation) -> str:
    return f"S{c.label}: {c.source_filename} — page {c.page_no} — chunk {c.ord}"
