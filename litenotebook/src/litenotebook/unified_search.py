"""Unified notebook search: sources, notes, chats and memory in one query.

Each section uses its own retrieval and keeps its own order; scores from
different systems are never mixed into one global ranking.

* Sources, lexical (default): FTS5/BM25 over chunks of ready sources. Never
  starts the embedding worker.
* Sources, hybrid (explicit): the Phase 1 retriever (FTS5 + FAISS + RRF). This
  needs a query embedding, so it may start the MiniLM worker (~15 s cold).
* Notes: FTS5 over title + content. Chats: FTS5 over message text and
  conversation titles (prompts are never stored, so never searched).
  Memory: the deterministic Phase 5 graph search.

Every query goes through quoted-term FTS syntax (operators, quotes and
brackets in user input are never interpreted) and is length-capped.
Snippets are escaped; only the application's own highlight markers become
<mark> elements.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from markupsafe import Markup, escape

from litenotebook.config import Settings
from litenotebook.retrieval.hybrid import fts_query

if TYPE_CHECKING:
    from litenotebook.embed.embedder import TextEmbedder

MAX_QUERY_CHARS = 500
MODES = ("all", "sources", "notes", "chats", "memory")
ENGINES = ("lexical", "hybrid")
PER_SECTION_ALL = 5
PER_SECTION = 20
_OPEN, _CLOSE = "\x02", "\x03"
SNIPPET_TOKENS = 18


@dataclass
class Hit:
    section: str
    title: str
    url: str
    snippet: Markup
    subtitle: str = ""
    panel_url: str | None = None  # HTMX evidence panel
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResults:
    query: str
    mode: str
    engine: str
    sections: dict[str, list[Hit]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.sections.values())


def clean_query(query: str) -> str:
    query = " ".join((query or "").replace("\x00", " ").split())
    query = query.replace(_OPEN, " ").replace(_CLOSE, " ")
    return query[:MAX_QUERY_CHARS]


def highlight(raw: str) -> Markup:
    """Escape everything, then turn only our own markers into <mark>."""
    escaped = str(escape(raw))
    return Markup(escaped.replace(_OPEN, "<mark>").replace(_CLOSE, "</mark>"))


def _clip(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rows(
    conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]
) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:  # defence in depth: fts_query is already safe
        return []


def search_sources_lexical(
    conn: sqlite3.Connection, notebook_id: int, query: str, limit: int
) -> list[Hit]:
    match = fts_query(query)
    if match is None:
        return []
    rows = _rows(
        conn,
        "SELECT c.id, c.page_no, c.ord, c.source_id, s.original_filename, "
        f"snippet(chunks_fts, 0, '{_OPEN}', '{_CLOSE}', '…', {SNIPPET_TOKENS}) "
        "AS snip, "
        "bm25(chunks_fts) AS score FROM chunks_fts "
        "JOIN chunks c ON c.id = chunks_fts.rowid JOIN sources s ON s.id = c.source_id "
        "WHERE chunks_fts MATCH ? AND c.notebook_id = ? AND s.notebook_id = ? "
        "AND s.status = 'ready' ORDER BY score LIMIT ?",
        (match, notebook_id, notebook_id, limit),
    )
    return [
        Hit(
            section="sources",
            title=r["original_filename"],
            subtitle=f"page {r['page_no']} · chunk {r['ord']}",
            url=f"/notebooks/{notebook_id}/chat/chunks/{r['id']}",
            panel_url=f"/notebooks/{notebook_id}/chat/chunks/{r['id']}",
            snippet=highlight(r["snip"]),
            meta={
                "match": "lexical",
                "bm25": round(r["score"], 3),
                "source_id": r["source_id"],
                "chunk_id": r["id"],
            },
        )
        for r in rows
    ]


def search_sources_hybrid(
    conn: sqlite3.Connection,
    settings: Settings,
    embedder: TextEmbedder,
    notebook_id: int,
    query: str,
    limit: int,
) -> list[Hit]:
    from litenotebook.index.faiss_store import FaissStore
    from litenotebook.retrieval.hybrid import Retriever

    results = Retriever(conn, settings, embedder, FaissStore(conn, settings)).search(
        notebook_id, query, limit
    )
    hits = []
    for r in results:
        kinds = [
            k
            for k, rank in (("semantic", r.semantic_rank), ("lexical", r.lexical_rank))
            if rank
        ]
        hits.append(
            Hit(
                section="sources",
                title=r.source_filename,
                subtitle=f"page {r.page_no} · chunk {r.ord}",
                url=f"/notebooks/{notebook_id}/chat/chunks/{r.chunk_id}",
                panel_url=f"/notebooks/{notebook_id}/chat/chunks/{r.chunk_id}",
                snippet=highlight(_clip(r.text)),
                meta={
                    "match": " + ".join(kinds) or "hybrid",
                    "fused": round(r.fused_score, 4),
                    "semantic_rank": r.semantic_rank,
                    "lexical_rank": r.lexical_rank,
                    "source_id": r.source_id,
                    "chunk_id": r.chunk_id,
                },
            )
        )
    return hits


def search_notes(
    conn: sqlite3.Connection, notebook_id: int, query: str, limit: int
) -> list[Hit]:
    match = fts_query(query)
    if match is None:
        return []
    rows = _rows(
        conn,
        "SELECT n.id, n.title, n.updated_at, "
        f"snippet(notes_fts, 1, '{_OPEN}', '{_CLOSE}', '…', {SNIPPET_TOKENS}) AS snip "
        "FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
        "WHERE notes_fts MATCH ? AND n.notebook_id = ? "
        "ORDER BY bm25(notes_fts, 5.0, 1.0), n.id DESC LIMIT ?",
        (match, notebook_id, limit),
    )
    return [
        Hit(
            section="notes",
            title=r["title"],
            subtitle=f"updated {r['updated_at'][:16].replace('T', ' ')}",
            url=f"/notebooks/{notebook_id}/notes/{r['id']}",
            snippet=highlight(r["snip"]) if r["snip"] else Markup(""),
            meta={"note_id": r["id"]},
        )
        for r in rows
    ]


def search_chats(
    conn: sqlite3.Connection, notebook_id: int, query: str, limit: int
) -> list[Hit]:
    match = fts_query(query)
    if match is None:
        return []
    hits: list[Hit] = []
    for r in _rows(
        conn,
        "SELECT c.id, c.title, c.updated_at FROM conversations_fts "
        "JOIN conversations c ON c.id = conversations_fts.rowid "
        "WHERE conversations_fts MATCH ? AND c.notebook_id = ? "
        "ORDER BY bm25(conversations_fts), c.id DESC LIMIT ?",
        (match, notebook_id, limit),
    ):
        hits.append(
            Hit(
                section="chats",
                title=r["title"],
                subtitle=f"conversation · {r['updated_at'][:16].replace('T', ' ')}",
                url=f"/notebooks/{notebook_id}/chat/{r['id']}",
                snippet=Markup(""),
                meta={"role": "conversation", "conversation_id": r["id"]},
            )
        )
    for r in _rows(
        conn,
        "SELECT m.id, m.role, m.created_at, c.id AS cid, c.title, "
        f"snippet(messages_fts, 0, '{_OPEN}', '{_CLOSE}', '…', {SNIPPET_TOKENS}) "
        "AS snip FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
        "JOIN conversations c ON c.id = m.conversation_id "
        "WHERE messages_fts MATCH ? AND c.notebook_id = ? "
        "AND m.status IN ('complete', 'interrupted') "
        "ORDER BY bm25(messages_fts), m.id DESC LIMIT ?",
        (match, notebook_id, limit),
    ):
        hits.append(
            Hit(
                section="chats",
                title=r["title"],
                subtitle=f"{'You' if r['role'] == 'user' else 'Assistant'} · "
                f"{r['created_at'][:16].replace('T', ' ')}",
                url=f"/notebooks/{notebook_id}/chat/{r['cid']}#msg-{r['id']}",
                snippet=highlight(r["snip"]),
                meta={"role": r["role"], "message_id": r["id"]},
            )
        )
    return hits[:limit]


def search_memory_section(
    conn: sqlite3.Connection, notebook_id: int, query: str, limit: int
) -> list[Hit]:
    from litenotebook.memory.search import search_memory

    found = search_memory(
        conn, notebook_id, query, include_dead_ends=True, limit=limit, char_budget=20000
    )
    return [
        Hit(
            section="memory",
            title=i.question,
            subtitle=f"{i.trust}"
            + (f" · {i.outcome.replace('_', ' ')}" if i.outcome else "")
            + f" · score {i.score:.3f}",
            url=f"/notebooks/{notebook_id}/memory/nodes/{i.node_id}",
            snippet=highlight(_clip(i.excerpt)),
            meta={"kind": i.kind, "score": i.score, "node_id": i.node_id},
        )
        for i in found.items
    ]


def unified_search(
    conn: sqlite3.Connection,
    settings: Settings,
    notebook_id: int,
    query: str,
    *,
    mode: str = "all",
    engine: str = "lexical",
    embedder: TextEmbedder | None = None,
) -> SearchResults:
    query = clean_query(query)
    mode = mode if mode in MODES else "all"
    engine = engine if engine in ENGINES else "lexical"
    results = SearchResults(query, mode, engine)
    if not query:
        return results
    limit = PER_SECTION_ALL if mode == "all" else PER_SECTION
    wanted = [m for m in MODES[1:] if mode in ("all", m)]
    for section in wanted:
        started = time.perf_counter()
        try:
            if section == "sources":
                if engine == "hybrid" and embedder is not None:
                    hits = search_sources_hybrid(
                        conn, settings, embedder, notebook_id, query, limit
                    )
                else:
                    hits = search_sources_lexical(conn, notebook_id, query, limit)
            elif section == "notes":
                hits = search_notes(conn, notebook_id, query, limit)
            elif section == "chats":
                hits = search_chats(conn, notebook_id, query, limit)
            else:
                hits = search_memory_section(conn, notebook_id, query, limit)
        except Exception as e:  # one section failing never breaks the others
            from litenotebook.embed.embedder import EmbeddingError

            results.errors[section] = (
                f"Semantic search is unavailable: {e}"
                if isinstance(e, EmbeddingError)
                else "This section could not be searched"
            )
            hits = []
        results.sections[section] = hits
        results.timings[section] = round(time.perf_counter() - started, 4)
    return results
