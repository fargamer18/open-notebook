"""Notebook-scoped hybrid retrieval (ported from LocalBook's ask.py).

FAISS (semantic) and FTS5/BM25 (lexical) candidates are fused with Reciprocal
Rank Fusion, near-duplicates are dropped, and the top k are returned. Every
query is restricted to exactly one notebook and to its ``ready`` sources (FAISS
candidates from any other source are dropped when rows are fetched).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from litenotebook.config import Settings
from litenotebook.embed.embedder import TextEmbedder
from litenotebook.index.faiss_store import FaissStore

log = logging.getLogger(__name__)

STOP_WORDS = frozenset(
    "a an and are as at be by can could do does explain for from how i in is it "
    "me of on or should tell than that the this to was were what when where which "
    "who why will with would you".split()
)
MAX_FTS_TERMS = 32
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class SearchResult:
    chunk_id: int
    source_id: int
    source_filename: str
    page_no: int
    ord: int
    char_start: int
    char_end: int
    text: str
    fused_score: float
    semantic_rank: int | None = None
    semantic_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None  # bm25(): lower is better

    def snippet(self, width: int = 240) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= width else flat[: width - 1] + "…"


def fts_query(text: str) -> str | None:
    """Turn arbitrary user text into a safe FTS5 OR-query of quoted terms.

    Only ``\\w+`` runs survive, each wrapped in double quotes, so FTS5 operators,
    column filters, quotes and parentheses in user input are never interpreted.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    terms = [w for w in words if w not in STOP_WORDS] or words
    unique = list(dict.fromkeys(terms))[:MAX_FTS_TERMS]
    if not unique:
        return None
    return " OR ".join(f'"{t}"' for t in unique)


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_d), ranks from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _normalized_words(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in _WORD.findall(text))


def dedupe(results: Sequence[SearchResult], threshold: float) -> list[SearchResult]:
    """Drop results too similar (word-set Jaccard > threshold) to a kept one."""
    kept: list[SearchResult] = []
    seen: list[frozenset[str]] = []
    for result in results:
        words = _normalized_words(result.text)
        if not words:
            continue
        if any(len(words & s) / len(words | s) > threshold for s in seen):
            continue
        kept.append(result)
        seen.append(words)
    return kept


class Retriever:
    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        embedder: TextEmbedder,
        store: FaissStore,
    ) -> None:
        self.conn = conn
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def lexical(
        self, notebook_id: int, query: str, limit: int
    ) -> list[tuple[int, float]]:
        match = fts_query(query)
        if match is None:
            return []
        try:
            rows = self.conn.execute(
                "SELECT c.id, bm25(chunks_fts) AS score FROM chunks_fts "
                "JOIN chunks c ON c.id = chunks_fts.rowid "
                "JOIN sources s ON s.id = c.source_id "
                "WHERE chunks_fts MATCH ? AND c.notebook_id = ? "
                "AND s.status = 'ready' "
                "ORDER BY score LIMIT ?",
                (match, notebook_id, limit),
            ).fetchall()
        except sqlite3.OperationalError as e:  # defense in depth; fts_query is safe
            log.warning(
                "FTS5 query failed (%s); lexical results skipped", type(e).__name__
            )
            return []
        return [(r["id"], r["score"]) for r in rows]

    def semantic(
        self, notebook_id: int, query: str, limit: int
    ) -> list[tuple[int, float]]:
        vector = self.embedder.encode([query])[0]
        return self.store.search(notebook_id, vector, limit)

    def has_ready_sources(self, notebook_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sources WHERE notebook_id = ? AND status = 'ready' LIMIT 1",
            (notebook_id,),
        ).fetchone()
        return row is not None

    def search(
        self, notebook_id: int, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        top_k = top_k or self.settings.retrieval_top_k
        limit = self.settings.retrieval_candidates
        semantic = self.semantic(notebook_id, query, limit)
        lexical = self.lexical(notebook_id, query, limit)

        fused = rrf_fuse(
            [[i for i, _ in semantic], [i for i, _ in lexical]], k=self.settings.rrf_k
        )
        if not fused:
            return []
        rows = self.conn.execute(
            "SELECT c.id, c.source_id, c.page_no, c.ord, c.char_start, c.char_end, "
            "c.text, s.original_filename FROM chunks c "
            "JOIN sources s ON s.id = c.source_id "
            "WHERE c.notebook_id = ? AND s.notebook_id = ? AND s.status = 'ready' "
            "AND c.id IN (SELECT value FROM json_each(?))",
            (notebook_id, notebook_id, "[" + ",".join(str(i) for i in fused) + "]"),
        ).fetchall()
        by_id = {r["id"]: r for r in rows}  # enforces notebook + ready scope

        sem = {cid: (rank, score) for rank, (cid, score) in enumerate(semantic, 1)}
        lex = {cid: (rank, score) for rank, (cid, score) in enumerate(lexical, 1)}
        ranked: list[SearchResult] = []
        for cid, fused_score in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0])):
            row = by_id.get(cid)
            if row is None:
                continue
            ranked.append(
                SearchResult(
                    chunk_id=cid,
                    source_id=row["source_id"],
                    source_filename=row["original_filename"],
                    page_no=row["page_no"],
                    ord=row["ord"],
                    char_start=row["char_start"],
                    char_end=row["char_end"],
                    text=row["text"],
                    fused_score=fused_score,
                    semantic_rank=sem[cid][0] if cid in sem else None,
                    semantic_score=sem[cid][1] if cid in sem else None,
                    lexical_rank=lex[cid][0] if cid in lex else None,
                    lexical_score=lex[cid][1] if cid in lex else None,
                )
            )
        return dedupe(ranked, self.settings.dedup_jaccard_threshold)[:top_k]
