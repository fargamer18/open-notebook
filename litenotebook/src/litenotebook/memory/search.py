"""Deterministic graph-memory retrieval (no MiniLM, no LLM).

    question → meaningful terms → FTS5 over memory nodes (≤ 5 seeds)
             → breadth-first walk over memory_edges, ≤ 2 hops (recursive CTE)
             → map reached nodes to remembered items → rank → ≤ 6 items

Remembered items:
  * exchange    – an earlier question with its model answer
  * correction  – the user's correction of an earlier answer
  * question    – an earlier question that never got a model answer

Score of an item (all terms deterministic; relevance always comes first):

    proximity = max over paths of seed_strength × 0.6^hops,
                seed_strength = 1 / (1 + seed rank)              (0 … 1)
    score     = proximity × trust × outcome  +  0.001 × id / newest id

    trust   : user correction 2.0 · source-backed answer 1.3 ·
              user question 1.1 · uncited assistant answer 1.0
    outcome : useful 1.4 · corrected (the superseded answer) 0.3 ·
              dead end 0.1 — and dead ends are left out of chat context
    The last term is a recency tie-breaker only.

So a correction strongly outranks the answer it corrects, but cannot
outrank a clearly more relevant memory. Ties: newest first.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from litenotebook.memory.concepts import terms

MAX_SEEDS = 5
MAX_DEPTH = 2
MAX_RESULTS = 6
MAX_WALK_ROWS = 400
CHAR_BUDGET = 2400
HOP_DECAY = 0.6
QUESTION_CHARS = 240
EXCERPT_CHARS = 420

TRUST_WEIGHT = {
    "correction": 2.0,
    "source_backed": 1.3,
    "user_note": 1.2,  # user-authored; below source evidence
    "user_statement": 1.1,
    "assistant_response": 1.0,
}
OUTCOME_WEIGHT = {"useful": 1.4, "corrected": 0.3, "dead_end": 0.1}
MAX_FTS_TERMS = 16


@dataclass
class MemoryItem:
    kind: str  # 'exchange' | 'correction' | 'question' | 'note'
    node_id: int
    node_key: str
    message_id: int  # assistant answer / user question id (0 for a note)
    provenance: str
    conversation_id: int | None
    conversation_title: str
    question: str
    excerpt: str
    outcome: str | None
    correction: str | None
    sources: list[dict[str, Any]]
    created_at: str
    score: float = 0.0
    hops: int = 0
    path: str = ""
    rank: int = 0
    note_id: int | None = None

    @property
    def source_backed(self) -> bool:
        return bool(self.sources)

    @property
    def trust(self) -> str:
        if self.kind == "correction":
            return "user correction"
        if self.kind == "question":
            return "user statement"
        if self.kind == "note":
            return "user note"
        return "source-backed answer" if self.sources else "uncited assistant answer"


@dataclass
class MemorySearch:
    items: list[MemoryItem]
    seeds: list[int] = field(default_factory=list)
    reached: int = 0
    timings: dict[str, float] = field(default_factory=dict)


def fts_match(query: str) -> str | None:
    words = terms(query)[:MAX_FTS_TERMS]
    return " OR ".join(f'"{w}"' for w in words) if words else None


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def search_memory(
    conn: sqlite3.Connection,
    notebook_id: int,
    query: str,
    *,
    exclude_message_ids: set[int] | frozenset[int] = frozenset(),
    include_dead_ends: bool = False,
    limit: int = MAX_RESULTS,
    max_depth: int = MAX_DEPTH,
    char_budget: int = CHAR_BUDGET,
) -> MemorySearch:
    started = time.perf_counter()
    match = fts_match(query)
    if match is None:
        return MemorySearch([])
    try:
        seeds = [
            int(r[0])
            for r in conn.execute(
                "SELECT n.id FROM memory_nodes_fts f JOIN memory_nodes n "
                "ON n.id = f.rowid WHERE memory_nodes_fts MATCH ? "
                "AND n.notebook_id = ? ORDER BY bm25(memory_nodes_fts, 4.0, 1.0), n.id "
                "LIMIT ?",
                (match, notebook_id, MAX_SEEDS),
            )
        ]
    except sqlite3.OperationalError:
        seeds = []
    seeded = time.perf_counter()
    if not seeds:
        return MemorySearch([], timings={"seed": seeded - started})

    walk = conn.execute(
        "WITH RECURSIVE walk(node_id, seed, depth) AS ("
        "  SELECT value, value, 0 FROM json_each(?) "
        "  UNION "
        "  SELECT CASE WHEN e.from_node_id = w.node_id THEN e.to_node_id "
        "              ELSE e.from_node_id END, w.seed, w.depth + 1 "
        "  FROM walk w JOIN memory_edges e "
        "    ON e.from_node_id = w.node_id OR e.to_node_id = w.node_id "
        "  WHERE w.depth < ? AND e.notebook_id = ? "
        "  LIMIT ?"
        ") "
        "SELECT w.node_id, w.seed, min(w.depth) AS hops, n.node_type, n.key, "
        "n.message_id, n.note_id FROM walk w JOIN memory_nodes n ON n.id = w.node_id "
        "WHERE n.notebook_id = ? GROUP BY w.node_id, w.seed",
        (json.dumps(seeds), max_depth, notebook_id, MAX_WALK_ROWS, notebook_id),
    ).fetchall()
    walked = time.perf_counter()

    strength = {seed: 1.0 / (1 + i) for i, seed in enumerate(seeds)}
    # anchor: the assistant answer id (or user question id) a node belongs to
    best: dict[tuple[str, int], tuple[float, int, str]] = {}
    for row in walk:
        node_type, message_id = row["node_type"], row["message_id"]
        if node_type in ("concept", "conversation"):
            continue
        proximity = strength[row["seed"]] * HOP_DECAY ** row["hops"]
        if row["note_id"] is not None:  # a note, or evidence linked from it
            anchor = ("note", int(row["note_id"]))
        elif message_id is None:
            continue
        elif node_type == "question":
            anchor = ("question", int(message_id))
        elif node_type == "correction":
            anchor = ("correction", int(message_id))
        else:  # answer, source_evidence
            anchor = ("answer", int(message_id))
        seed_label = row["key"] if row["seed"] == row["node_id"] else ""
        if anchor not in best or proximity > best[anchor][0]:
            best[anchor] = (proximity, row["hops"], seed_label or str(row["seed"]))

    # questions with an answer are represented by that answer
    answered = {
        int(r[0]): int(r[1])
        for r in conn.execute(
            "SELECT reply_to, id FROM messages WHERE role = 'assistant' "
            "AND reply_to IN (SELECT value FROM json_each(?))",
            (json.dumps([a[1] for a in best if a[0] == "question"]),),
        )
    }
    merged: dict[tuple[str, int], tuple[float, int, str]] = {}
    for (kind, mid), value in best.items():
        if kind == "question" and mid in answered:
            target = ("answer", answered[mid])
            if not _node_exists(conn, notebook_id, f"answer:{answered[mid]}"):
                target = ("question", mid)
        else:
            target = (kind, mid)
        if target not in merged or value[0] > merged[target][0]:
            merged[target] = value

    items = _load_items(conn, notebook_id, merged, exclude_message_ids)
    newest = max((i.node_id for i in items), default=1) or 1
    kept: list[MemoryItem] = []
    for item in items:
        if item.kind == "exchange" and item.outcome == "dead_end":
            if not include_dead_ends:
                continue
        trust = TRUST_WEIGHT[
            "correction" if item.kind == "correction" else item.provenance
        ]
        outcome = (
            OUTCOME_WEIGHT[item.outcome]
            if item.kind == "exchange" and item.outcome
            else 1.0
        )
        recency = 0.001 * item.node_id / newest
        item.score = round(item.score * trust * outcome + recency, 4)
        kept.append(item)
    kept.sort(key=lambda i: (-i.score, -i.node_id))

    results: list[MemoryItem] = []
    used = 0
    for item in kept:
        size = len(item.question) + len(item.excerpt) + len(item.correction or "")
        if len(results) >= limit or (results and used + size > char_budget):
            break
        used += size
        item.rank = len(results) + 1
        results.append(item)
    done = time.perf_counter()
    return MemorySearch(
        results,
        seeds=seeds,
        reached=len(walk),
        timings={
            "seed": round(seeded - started, 5),
            "traverse": round(walked - seeded, 5),
            "rank": round(done - walked, 5),
            "total": round(done - started, 5),
        },
    )


def _node_exists(conn: sqlite3.Connection, notebook_id: int, key: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM memory_nodes WHERE notebook_id = ? AND key = ?",
            (notebook_id, key),
        ).fetchone()
        is not None
    )


def _sources(conn: sqlite3.Connection, message_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT mc.source_filename, mc.page_no, mc.chunk_ord, mc.chunk_id, "
        "mc.source_id, EXISTS (SELECT 1 FROM chunks ch WHERE ch.id = mc.chunk_id "
        "AND ch.source_id = mc.source_id) AS live "
        "FROM message_citations mc WHERE mc.message_id = ? ORDER BY mc.ordinal",
        (message_id,),
    ).fetchall()
    return [
        {
            "filename": r["source_filename"],
            "page": r["page_no"],
            "chunk": r["chunk_ord"],
            "chunk_id": r["chunk_id"],
            "source_id": r["source_id"],
            "deleted": not r["live"],
        }
        for r in rows
    ]


def refresh_source_state(
    conn: sqlite3.Connection, sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Re-check whether each cited chunk still exists in its original source
    (snapshots keep the identity; deletion is decided now, never relinked)."""
    for src in sources:
        chunk_id, source_id = src.get("chunk_id"), src.get("source_id")
        if isinstance(chunk_id, int) and isinstance(source_id, int):
            live = conn.execute(
                "SELECT 1 FROM chunks WHERE id = ? AND source_id = ?",
                (chunk_id, source_id),
            ).fetchone()
            src["deleted"] = live is None
    return sources


def _load_items(
    conn: sqlite3.Connection,
    notebook_id: int,
    anchors: dict[tuple[str, int], tuple[float, int, str]],
    exclude: set[int] | frozenset[int],
) -> list[MemoryItem]:
    items: list[MemoryItem] = []
    note_rows = _note_rows(conn, notebook_id, [m for k, m in anchors if k == "note"])
    for (kind, mid), (proximity, hops, path) in anchors.items():
        if kind == "note":
            note = _note_item(note_rows.get(mid), mid, proximity, hops, path)
            if note is not None:
                items.append(note)
            continue
        if mid in exclude and kind != "correction":
            continue  # already in the recent-chat window (corrections never are)
        key = {"answer": "answer", "correction": "correction", "question": "question"}[
            kind
        ] + f":{mid}"
        node = conn.execute(
            "SELECT n.id, n.key, n.provenance, n.content, n.created_at, "
            "n.conversation_id, c.title FROM memory_nodes n "
            "LEFT JOIN conversations c ON c.id = n.conversation_id "
            "WHERE n.notebook_id = ? AND n.key = ?",
            (notebook_id, key),
        ).fetchone()
        if node is None:
            continue
        if kind == "question":
            question, answer_id, outcome, correction = node["content"], mid, None, None
            excerpt = ""
        else:
            row = conn.execute(
                "SELECT a.content AS answer, u.content AS question, o.outcome, "
                "o.correction FROM messages a JOIN messages u ON u.id = a.reply_to "
                "LEFT JOIN memory_outcomes o ON o.message_id = a.id WHERE a.id = ?",
                (mid,),
            ).fetchone()
            if row is None:
                continue
            question, answer_id = row["question"], mid
            outcome, correction = row["outcome"], row["correction"]
            excerpt = row["answer"] if kind == "answer" else row["correction"] or ""
        items.append(
            MemoryItem(
                kind="exchange" if kind == "answer" else kind,
                node_id=node["id"],
                node_key=node["key"],
                message_id=answer_id,
                provenance=node["provenance"],
                conversation_id=node["conversation_id"],
                conversation_title=node["title"] or "(deleted conversation)",
                question=_clip(question, QUESTION_CHARS),
                excerpt=_clip(excerpt, EXCERPT_CHARS),
                outcome=outcome,
                correction=correction,
                sources=_sources(conn, answer_id) if kind != "question" else [],
                created_at=node["created_at"],
                score=proximity,
                hops=hops,
                path=path,
            )
        )
    return items


FILTERS = ("all", "useful", "corrections", "dead_ends", "notes")


def list_items(
    conn: sqlite3.Connection, notebook_id: int, which: str = "all", limit: int = 30
) -> list[MemoryItem]:
    """Recent remembered items for the Memory page (newest first)."""
    if which == "useful":
        sql = (
            "SELECT 'answer', o.message_id FROM memory_outcomes o "
            "WHERE o.notebook_id = ? AND o.outcome = 'useful' "
            "ORDER BY o.updated_at DESC, o.id DESC LIMIT ?"
        )
    elif which == "dead_ends":
        sql = (
            "SELECT 'answer', o.message_id FROM memory_outcomes o "
            "WHERE o.notebook_id = ? AND o.outcome = 'dead_end' "
            "ORDER BY o.updated_at DESC, o.id DESC LIMIT ?"
        )
    elif which == "notes":
        sql = (
            "SELECT 'note', n.note_id FROM memory_nodes n "
            "WHERE n.notebook_id = ? AND n.node_type = 'note' "
            "ORDER BY n.updated_at DESC, n.id DESC LIMIT ?"
        )
    elif which == "corrections":
        sql = (
            "SELECT 'correction', n.message_id FROM memory_nodes n "
            "WHERE n.notebook_id = ? AND n.node_type = 'correction' "
            "ORDER BY n.updated_at DESC, n.id DESC LIMIT ?"
        )
    else:
        sql = (
            "SELECT CASE n.node_type WHEN 'answer' THEN 'answer' "
            "WHEN 'correction' THEN 'correction' WHEN 'note' THEN 'note' "
            "ELSE 'question' END, coalesce(n.message_id, n.note_id) "
            "FROM memory_nodes n WHERE n.notebook_id = ? AND ("
            "  n.node_type IN ('answer', 'correction', 'note') "
            "  OR (n.node_type = 'question' "
            "  AND NOT EXISTS (SELECT 1 FROM messages a JOIN memory_nodes x "
            "  ON x.notebook_id = n.notebook_id AND x.key = 'answer:' || a.id "
            "  WHERE a.reply_to = n.message_id))) "
            "ORDER BY n.id DESC LIMIT ?"
        )
    anchors = {
        (str(r[0]), int(r[1])): (0.0, 0, "")
        for r in conn.execute(sql, (notebook_id, limit))
        if r[1] is not None
    }
    items = _load_items(conn, notebook_id, anchors, frozenset())
    items.sort(key=lambda i: -i.node_id)
    for rank, item in enumerate(items, 1):
        item.rank = rank
    return items


@dataclass
class Neighbour:
    node_id: int
    node_type: str
    label: str
    relation: str
    direction: str  # 'out' (this → neighbour) or 'in' (neighbour → this)


@dataclass
class NodeDetail:
    id: int
    key: str
    node_type: str
    label: str
    content: str
    provenance: str
    conversation_id: int | None
    conversation_title: str | None
    message_id: int | None
    source_id: int | None
    chunk_id: int | None
    note_id: int | None
    source_exists: bool
    created_at: str
    updated_at: str
    neighbours: list[Neighbour]


def node_detail(
    conn: sqlite3.Connection, notebook_id: int, node_id: int, max_neighbours: int = 60
) -> NodeDetail | None:
    row = conn.execute(
        "SELECT n.*, c.title AS conversation_title, "
        "(n.source_id IS NOT NULL AND EXISTS (SELECT 1 FROM chunks ch "
        " WHERE ch.id = n.chunk_id AND ch.source_id = n.source_id)) AS live "
        "FROM memory_nodes n LEFT JOIN conversations c ON c.id = n.conversation_id "
        "WHERE n.id = ? AND n.notebook_id = ?",
        (node_id, notebook_id),
    ).fetchone()
    if row is None:
        return None
    neighbours = [
        Neighbour(r["id"], r["node_type"], r["label"], r["relation"], r["dir"])
        for r in conn.execute(
            "SELECT n.id, n.node_type, n.label, e.relation, 'out' AS dir "
            "FROM memory_edges e JOIN memory_nodes n ON n.id = e.to_node_id "
            "WHERE e.from_node_id = ? AND e.notebook_id = ? "
            "UNION ALL "
            "SELECT n.id, n.node_type, n.label, e.relation, 'in' AS dir "
            "FROM memory_edges e JOIN memory_nodes n ON n.id = e.from_node_id "
            "WHERE e.to_node_id = ? AND e.notebook_id = ? "
            "ORDER BY 4, 2, 1 LIMIT ?",
            (node_id, notebook_id, node_id, notebook_id, max_neighbours),
        )
    ]
    return NodeDetail(
        id=row["id"],
        key=row["key"],
        node_type=row["node_type"],
        label=row["label"],
        content=row["content"],
        provenance=row["provenance"],
        conversation_id=row["conversation_id"],
        conversation_title=row["conversation_title"],
        message_id=row["message_id"],
        source_id=row["source_id"],
        chunk_id=row["chunk_id"],
        note_id=row["note_id"],
        source_exists=bool(row["live"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        neighbours=neighbours,
    )


def node_id_for_key(conn: sqlite3.Connection, notebook_id: int, key: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM memory_nodes WHERE notebook_id = ? AND key = ?",
        (notebook_id, key),
    ).fetchone()
    return int(row[0]) if row else None


def _note_rows(
    conn: sqlite3.Connection, notebook_id: int, note_ids: list[int]
) -> dict[int, sqlite3.Row]:
    """Memory node + note for many notes at once (no query per note)."""
    found: dict[int, sqlite3.Row] = {}
    for start in range(0, len(note_ids), 500):
        batch = note_ids[start : start + 500]
        marks = ",".join("?" * len(batch))
        for row in conn.execute(
            "SELECT m.id, m.key, m.created_at, n.id AS note_id, n.title, n.content "
            "FROM memory_nodes m JOIN notes n ON n.id = m.note_id "
            f"WHERE m.notebook_id = ? AND m.key IN ({marks})",
            (notebook_id, *(f"note:{i}" for i in batch)),
        ):
            found[int(row["note_id"])] = row
    return found


def _note_item(
    row: sqlite3.Row | None,
    note_id: int,
    proximity: float,
    hops: int,
    path: str,
) -> MemoryItem | None:
    if row is None:
        return None
    return MemoryItem(
        kind="note",
        node_id=row["id"],
        node_key=row["key"],
        message_id=0,
        provenance="user_note",
        conversation_id=None,
        conversation_title=row["title"],
        question=_clip(row["title"], QUESTION_CHARS),
        excerpt=_clip(row["content"], EXCERPT_CHARS),
        outcome=None,
        correction=None,
        sources=[],
        created_at=row["created_at"],
        score=proximity,
        hops=hops,
        path=path,
        note_id=note_id,
    )
