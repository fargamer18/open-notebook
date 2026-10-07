"""Parse ``[S#]`` citations and resolve them against the supplied context.

Source/page/chunk metadata always comes from the retrieval results, never from
the model's text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from litenotebook.retrieval.hybrid import SearchResult

# [S1]  [S1, S2]  [S1,S2; S3]  [S1, 2]  (the S may be omitted after the first)
_GROUP = re.compile(r"\[\s*S\s*(\d+)((?:\s*[,;]\s*S?\s*\d+)*)\s*\]", re.IGNORECASE)
_NUMBER = re.compile(r"\d+")


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


def parse_labels(text: str) -> list[int]:
    """All cited labels in order of first appearance (duplicates removed)."""
    labels: list[int] = []
    for match in _GROUP.finditer(text):
        for number in [match.group(1), *_NUMBER.findall(match.group(2))]:
            label = int(number)
            if label not in labels:
                labels.append(label)
    return labels


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
