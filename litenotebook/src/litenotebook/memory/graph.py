"""Build the memory graph deterministically from authoritative chat data.

For one exchange (a user question and its assistant answer)::

    conversation ─contains→ question ─answered_by→ answer ─supported_by→ evidence
                  └contains→ answer
    previous question ─followed_by→ question
    question / answer / correction ─mentions→ concept
    correction ─corrects→ answer

Every edge an exchange creates carries ``provenance = 'message:<answer id>'``,
so re-syncing deletes and recreates exactly that exchange's edges. Nodes are
upserted by a deterministic key, so ids are stable and nothing duplicates.
Concept nodes are shared per notebook and pruned when nothing mentions them.

No LLM is ever called here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from litenotebook.memory.concepts import extract_concepts

NODE_TYPES = frozenset(
    {
        "conversation",
        "question",
        "answer",
        "concept",
        "source_evidence",
        "correction",
        "note",
    }
)
RELATIONS = frozenset(
    {
        "contains",
        "answered_by",
        "followed_by",
        "mentions",
        "supported_by",
        "corrects",
        "note_mentions",
        "note_supported_by",
    }
)
NOTE_CONCEPTS = 8
NOTE_CONCEPT_CHARS = 20000  # concepts come from the start of very long notes
OUTCOMES = ("useful", "dead_end", "corrected")
MAX_CORRECTION_CHARS = 2000
CONCEPTS_PER_TEXT = 6
NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


class MemoryGraphError(ValueError):
    pass


class OutcomeError(MemoryGraphError):
    pass


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _upsert_node(
    conn: sqlite3.Connection,
    notebook_id: int,
    key: str,
    node_type: str,
    label: str,
    content: str,
    provenance: str,
    *,
    conversation_id: int | None = None,
    message_id: int | None = None,
    source_id: int | None = None,
    chunk_id: int | None = None,
    note_id: int | None = None,
) -> int:
    assert node_type in NODE_TYPES
    conn.execute(
        "INSERT INTO memory_nodes (notebook_id, key, node_type, label, content, "
        "provenance, conversation_id, message_id, source_id, chunk_id, note_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(notebook_id, key) DO UPDATE SET "
        "node_type = excluded.node_type, label = excluded.label, "
        "content = excluded.content, provenance = excluded.provenance, "
        "conversation_id = excluded.conversation_id, "
        "message_id = excluded.message_id, source_id = excluded.source_id, "
        "chunk_id = excluded.chunk_id, note_id = excluded.note_id, "
        f"updated_at = {NOW_SQL} "
        "WHERE memory_nodes.label IS NOT excluded.label "
        "OR memory_nodes.content IS NOT excluded.content "
        "OR memory_nodes.provenance IS NOT excluded.provenance "
        "OR memory_nodes.conversation_id IS NOT excluded.conversation_id",
        (
            notebook_id,
            key,
            node_type,
            label,
            content,
            provenance,
            conversation_id,
            message_id,
            source_id,
            chunk_id,
            note_id,
        ),
    )
    row = conn.execute(
        "SELECT id FROM memory_nodes WHERE notebook_id = ? AND key = ?",
        (notebook_id, key),
    ).fetchone()
    return int(row[0])


def _edge(
    conn: sqlite3.Connection,
    notebook_id: int,
    from_id: int,
    to_id: int,
    relation: str,
    provenance: str,
) -> None:
    if relation not in RELATIONS:  # the vocabulary is closed; never free text
        raise ValueError(f"Unknown relation {relation!r}")
    if from_id == to_id:
        return
    conn.execute(
        "INSERT OR IGNORE INTO memory_edges "
        "(notebook_id, from_node_id, to_node_id, relation, provenance) "
        "VALUES (?, ?, ?, ?, ?)",
        (notebook_id, from_id, to_id, relation, provenance),
    )


def _concepts(
    conn: sqlite3.Connection,
    notebook_id: int,
    text: str,
    limit: int = CONCEPTS_PER_TEXT,
) -> list[int]:
    return [
        _upsert_node(conn, notebook_id, f"concept:{c}", "concept", c, c, "derived")
        for c in extract_concepts(text, limit)
    ]


def _delete_keys(conn: sqlite3.Connection, notebook_id: int, keys: list[str]) -> None:
    conn.executemany(
        "DELETE FROM memory_nodes WHERE notebook_id = ? AND key = ?",
        [(notebook_id, k) for k in keys],
    )


def prune_orphan_concepts(conn: sqlite3.Connection, notebook_id: int) -> int:
    cur = conn.execute(
        "DELETE FROM memory_nodes WHERE notebook_id = ? AND node_type = 'concept' "
        "AND NOT EXISTS (SELECT 1 FROM memory_edges e "
        "WHERE e.from_node_id = memory_nodes.id OR e.to_node_id = memory_nodes.id)",
        (notebook_id,),
    )
    return cur.rowcount


def sync_turn(conn: sqlite3.Connection, assistant_message_id: int) -> bool:
    """(Re)derive the graph for one exchange. Idempotent. Returns False if the
    message is not an assistant answer."""
    row = conn.execute(
        "SELECT a.id AS aid, a.status, a.content AS answer, a.model, a.reply_to, "
        "u.content AS question, c.id AS cid, c.title, c.notebook_id, "
        "o.outcome, o.correction "
        "FROM messages a JOIN messages u ON u.id = a.reply_to "
        "JOIN conversations c ON c.id = a.conversation_id "
        "LEFT JOIN memory_outcomes o ON o.message_id = a.id "
        "WHERE a.id = ? AND a.role = 'assistant'",
        (assistant_message_id,),
    ).fetchone()
    if row is None:
        return False
    nb, aid, uid, cid = row["notebook_id"], row["aid"], row["reply_to"], row["cid"]
    prov = f"message:{aid}"
    answered = row["status"] == "complete" and row["model"] is not None
    citations = conn.execute(
        "SELECT * FROM message_citations WHERE message_id = ? ORDER BY ordinal",
        (aid,),
    ).fetchall()
    with conn:
        conn.execute("DELETE FROM memory_edges WHERE provenance = ?", (prov,))
        conv = _upsert_node(
            conn,
            nb,
            f"conversation:{cid}",
            "conversation",
            row["title"],
            row["title"],
            "derived",
            conversation_id=cid,
        )
        q = _upsert_node(
            conn,
            nb,
            f"question:{uid}",
            "question",
            _clip(row["question"], 120),
            row["question"],
            "user_statement",
            conversation_id=cid,
            message_id=uid,
        )
        _edge(conn, nb, conv, q, "contains", prov)
        previous = conn.execute(
            "SELECT n.id FROM messages m JOIN memory_nodes n "
            "ON n.notebook_id = ? AND n.key = 'question:' || m.id "
            "WHERE m.conversation_id = ? AND m.role = 'user' AND m.id < ? "
            "ORDER BY m.id DESC LIMIT 1",
            (nb, cid, uid),
        ).fetchone()
        if previous is not None:
            _edge(conn, nb, int(previous[0]), q, "followed_by", prov)
        for concept in _concepts(conn, nb, f"{row['question']}"):
            _edge(conn, nb, q, concept, "mentions", prov)

        evidence_keys = [f"evidence:{c['id']}" for c in citations]
        stale = conn.execute(
            "SELECT key FROM memory_nodes WHERE notebook_id = ? AND message_id = ? "
            "AND node_type = 'source_evidence'",
            (nb, aid),
        ).fetchall()
        _delete_keys(
            conn, nb, [r[0] for r in stale if not answered or r[0] not in evidence_keys]
        )
        if answered:
            a = _upsert_node(
                conn,
                nb,
                f"answer:{aid}",
                "answer",
                _clip(row["answer"], 120),
                row["answer"],
                "source_backed" if citations else "assistant_response",
                conversation_id=cid,
                message_id=aid,
            )
            _edge(conn, nb, conv, a, "contains", prov)
            _edge(conn, nb, q, a, "answered_by", prov)
            for concept in _concepts(conn, nb, row["answer"]):
                _edge(conn, nb, a, concept, "mentions", prov)
            for c in citations:
                ev = _upsert_node(
                    conn,
                    nb,
                    f"evidence:{c['id']}",
                    "source_evidence",
                    f"{c['source_filename']} — page {c['page_no']}",
                    c["chunk_text"],
                    "source_backed",
                    conversation_id=cid,
                    message_id=aid,
                    source_id=c["source_id"],
                    chunk_id=c["chunk_id"],
                )
                _edge(conn, nb, a, ev, "supported_by", prov)
        else:
            _delete_keys(conn, nb, [f"answer:{aid}"])

        if answered and row["outcome"] == "corrected" and row["correction"]:
            fix = _upsert_node(
                conn,
                nb,
                f"correction:{aid}",
                "correction",
                _clip(row["correction"], 120),
                row["correction"],
                "user_correction",
                conversation_id=cid,
                message_id=aid,
            )
            _edge(conn, nb, fix, a, "corrects", prov)
            _edge(conn, nb, q, fix, "answered_by", prov)
            for concept in _concepts(conn, nb, row["correction"]):
                _edge(conn, nb, fix, concept, "mentions", prov)
        else:
            _delete_keys(conn, nb, [f"correction:{aid}"])
        prune_orphan_concepts(conn, nb)
    return True


def sync_note(conn: sqlite3.Connection, note_id: int) -> bool:
    """(Re)derive a note's graph: note ─note_mentions→ concepts and
    note ─note_supported_by→ linked evidence / answers / memory. Idempotent;
    every edge carries provenance 'note:<id>'."""
    row = conn.execute(
        "SELECT id, notebook_id, title, content FROM notes WHERE id = ?", (note_id,)
    ).fetchone()
    if row is None:
        return False
    nb, prov = row["notebook_id"], f"note:{note_id}"
    links = conn.execute(
        "SELECT * FROM note_links WHERE note_id = ? ORDER BY id", (note_id,)
    ).fetchall()
    with conn:
        conn.execute("DELETE FROM memory_edges WHERE provenance = ?", (prov,))
        n = _upsert_node(
            conn,
            nb,
            f"note:{note_id}",
            "note",
            row["title"],
            row["content"][:4000],
            "user_note",
            note_id=note_id,
        )
        text = f"{row['title']}\n{row['content'][:NOTE_CONCEPT_CHARS]}"
        for concept in _concepts(conn, nb, text, NOTE_CONCEPTS):
            _edge(conn, nb, n, concept, "note_mentions", prov)
        evidence_keys = []
        for link in links:
            target: int | None = None
            if link["link_type"] == "source_chunk":
                key = f"note_evidence:{link['id']}"
                evidence_keys.append(key)
                target = _upsert_node(
                    conn,
                    nb,
                    key,
                    "source_evidence",
                    f"{link['label']} — page {link['page_no']}",
                    link["excerpt"],
                    "source_backed",
                    note_id=note_id,
                    source_id=link["source_id"],
                    chunk_id=link["chunk_id"],
                )
            else:
                keys = (
                    (f"answer:{link['message_id']}", f"question:{link['message_id']}")
                    if link["link_type"] == "message"
                    else (link["memory_key"], link["memory_key"])
                )
                found = conn.execute(
                    "SELECT id FROM memory_nodes WHERE notebook_id = ? "
                    "AND key IN (?, ?) "
                    "ORDER BY node_type = 'answer' DESC LIMIT 1",
                    (nb, *keys),
                ).fetchone()
                target = int(found[0]) if found else None
            if target is not None:
                _edge(conn, nb, n, target, "note_supported_by", prov)
        stale = conn.execute(
            "SELECT key FROM memory_nodes WHERE notebook_id = ? AND note_id = ? "
            "AND node_type = 'source_evidence'",
            (nb, note_id),
        ).fetchall()
        _delete_keys(conn, nb, [r[0] for r in stale if r[0] not in evidence_keys])
        prune_orphan_concepts(conn, nb)
    return True


def sync_conversation(conn: sqlite3.Connection, conversation_id: int) -> int:
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM messages WHERE conversation_id = ? AND role = 'assistant' "
            "ORDER BY id",
            (conversation_id,),
        )
    ]
    for aid in ids:
        sync_turn(conn, aid)
    return len(ids)


def backfill(conn: sqlite3.Connection, notebook_id: int | None = None) -> int:
    """Sync exchanges that have no graph yet (startup). Idempotent and cheap."""
    rows = conn.execute(
        "SELECT a.id FROM messages a JOIN conversations c ON c.id = a.conversation_id "
        "WHERE a.role = 'assistant' AND a.status <> 'generating' "
        "AND (? IS NULL OR c.notebook_id = ?) "
        "AND NOT EXISTS (SELECT 1 FROM memory_nodes n "
        "    WHERE n.notebook_id = c.notebook_id "
        "    AND n.key = 'question:' || a.reply_to) "
        "ORDER BY a.id",
        (notebook_id, notebook_id),
    ).fetchall()
    for r in rows:
        sync_turn(conn, r[0])
    notes = conn.execute(
        "SELECT n.id FROM notes n WHERE (? IS NULL OR n.notebook_id = ?) "
        "AND NOT EXISTS (SELECT 1 FROM memory_nodes m "
        "    WHERE m.notebook_id = n.notebook_id AND m.key = 'note:' || n.id) "
        "ORDER BY n.id",
        (notebook_id, notebook_id),
    ).fetchall()
    for r in notes:
        sync_note(conn, r[0])
    return len(rows) + len(notes)


def rebuild(conn: sqlite3.Connection, notebook_id: int) -> int:
    """Drop and re-derive the notebook's graph. Outcomes/corrections are kept."""
    with conn:
        conn.execute("DELETE FROM memory_nodes WHERE notebook_id = ?", (notebook_id,))
    return backfill(conn, notebook_id)


# --- outcomes -------------------------------------------------------------------------


def set_outcome(
    conn: sqlite3.Connection,
    notebook_id: int,
    assistant_message_id: int,
    outcome: str | None,
    correction: str | None = None,
) -> None:
    """Useful / dead end / corrected (or None to clear). Never rewrites the
    original answer; the graph is re-synced so retrieval sees it at once."""
    row = conn.execute(
        "SELECT m.status, m.model FROM messages m "
        "JOIN conversations c ON c.id = m.conversation_id "
        "WHERE m.id = ? AND c.notebook_id = ? AND m.role = 'assistant'",
        (assistant_message_id, notebook_id),
    ).fetchone()
    if row is None:
        from litenotebook.chat import MessageNotFoundError

        raise MessageNotFoundError(
            f"Message {assistant_message_id} not found in this notebook"
        )
    if row["status"] != "complete" or row["model"] is None:
        raise OutcomeError("Only completed model answers can be rated")
    if outcome is not None and outcome not in OUTCOMES:
        raise OutcomeError("Unknown outcome")
    text = " ".join((correction or "").split()) if outcome == "corrected" else None
    if outcome == "corrected":
        if not text:
            raise OutcomeError("Enter the correction")
        if len(text) > MAX_CORRECTION_CHARS:
            raise OutcomeError(
                f"Corrections are limited to {MAX_CORRECTION_CHARS} characters"
            )
    with conn:
        if outcome is None:
            conn.execute(
                "DELETE FROM memory_outcomes WHERE message_id = ?",
                (assistant_message_id,),
            )
        else:
            conn.execute(
                "INSERT INTO memory_outcomes (notebook_id, message_id, outcome, "
                "correction) VALUES (?, ?, ?, ?) ON CONFLICT(message_id) DO UPDATE "
                "SET outcome = excluded.outcome, correction = excluded.correction, "
                f"updated_at = {NOW_SQL}",
                (notebook_id, assistant_message_id, outcome, text),
            )
    sync_turn(conn, assistant_message_id)


def get_outcome(
    conn: sqlite3.Connection, assistant_message_id: int
) -> tuple[str, str | None] | None:
    row = conn.execute(
        "SELECT outcome, correction FROM memory_outcomes WHERE message_id = ?",
        (assistant_message_id,),
    ).fetchone()
    return (row["outcome"], row["correction"]) if row else None


# --- statistics & consistency ------------------------------------------------------


@dataclass
class MemoryStats:
    nodes: dict[str, int] = field(default_factory=dict)
    edges: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, int] = field(default_factory=dict)
    exchanges: int = 0
    synced_exchanges: int = 0
    orphan_concepts: int = 0
    notes: int = 0
    synced_notes: int = 0
    fts_rows: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        return sum(self.nodes.values())

    @property
    def total_edges(self) -> int:
        return sum(self.edges.values())

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "outcomes": self.outcomes,
            "exchanges": self.exchanges,
            "synced_exchanges": self.synced_exchanges,
            "notes": self.notes,
            "synced_notes": self.synced_notes,
            "problems": self.problems,
        }


def stats(conn: sqlite3.Connection, notebook_id: int) -> MemoryStats:
    s = MemoryStats()
    s.nodes = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT node_type, count(*) FROM memory_nodes WHERE notebook_id = ? "
            "GROUP BY node_type",
            (notebook_id,),
        )
    }
    s.edges = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT relation, count(*) FROM memory_edges WHERE notebook_id = ? "
            "GROUP BY relation",
            (notebook_id,),
        )
    }
    s.outcomes = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT outcome, count(*) FROM memory_outcomes WHERE notebook_id = ? "
            "GROUP BY outcome",
            (notebook_id,),
        )
    }
    return s


def check(conn: sqlite3.Connection, notebook_id: int) -> MemoryStats:
    s = stats(conn, notebook_id)
    s.exchanges = conn.execute(
        "SELECT count(*) FROM messages a JOIN conversations c "
        "ON c.id = a.conversation_id WHERE c.notebook_id = ? AND a.role = 'assistant' "
        "AND a.status <> 'generating'",
        (notebook_id,),
    ).fetchone()[0]
    s.synced_exchanges = conn.execute(
        "SELECT count(*) FROM messages a JOIN conversations c "
        "ON c.id = a.conversation_id WHERE c.notebook_id = ? AND a.role = 'assistant' "
        "AND a.status <> 'generating' AND EXISTS (SELECT 1 FROM memory_nodes n "
        "WHERE n.notebook_id = c.notebook_id AND n.key = 'question:' || a.reply_to)",
        (notebook_id,),
    ).fetchone()[0]
    s.orphan_concepts = conn.execute(
        "SELECT count(*) FROM memory_nodes n WHERE n.notebook_id = ? "
        "AND n.node_type = 'concept' AND NOT EXISTS (SELECT 1 FROM memory_edges e "
        "WHERE e.from_node_id = n.id OR e.to_node_id = n.id)",
        (notebook_id,),
    ).fetchone()[0]
    cross = conn.execute(
        "SELECT count(*) FROM memory_edges e "
        "JOIN memory_nodes a ON a.id = e.from_node_id "
        "JOIN memory_nodes b ON b.id = e.to_node_id WHERE e.notebook_id = ? "
        "AND (a.notebook_id <> e.notebook_id OR b.notebook_id <> e.notebook_id)",
        (notebook_id,),
    ).fetchone()[0]
    s.fts_rows = conn.execute(
        "SELECT count(*) FROM memory_nodes_fts f JOIN memory_nodes n ON n.id = f.rowid "
        "WHERE n.notebook_id = ?",
        (notebook_id,),
    ).fetchone()[0]
    missing_corrections = conn.execute(
        "SELECT count(*) FROM memory_outcomes o JOIN messages m ON m.id = o.message_id "
        "WHERE o.notebook_id = ? AND o.outcome = 'corrected' AND m.status = 'complete' "
        "AND NOT EXISTS (SELECT 1 FROM memory_nodes n "
        "WHERE n.notebook_id = o.notebook_id "
        "AND n.key = 'correction:' || o.message_id)",
        (notebook_id,),
    ).fetchone()[0]
    s.notes = conn.execute(
        "SELECT count(*) FROM notes WHERE notebook_id = ?", (notebook_id,)
    ).fetchone()[0]
    s.synced_notes = conn.execute(
        "SELECT count(*) FROM notes n WHERE n.notebook_id = ? AND EXISTS ("
        "SELECT 1 FROM memory_nodes m WHERE m.notebook_id = n.notebook_id "
        "AND m.key = 'note:' || n.id)",
        (notebook_id,),
    ).fetchone()[0]
    if s.synced_notes != s.notes:
        s.problems.append(f"{s.notes - s.synced_notes} note(s) not in the graph")
    if s.synced_exchanges != s.exchanges:
        s.problems.append(
            f"{s.exchanges - s.synced_exchanges} exchange(s) not in the graph"
        )
    if s.orphan_concepts:
        s.problems.append(f"{s.orphan_concepts} unused concept node(s)")
    if cross:
        s.problems.append(f"{cross} edge(s) cross notebooks")
    if s.fts_rows != s.total_nodes:
        s.problems.append("memory search index out of step with nodes")
    if missing_corrections:
        s.problems.append(f"{missing_corrections} correction(s) missing from graph")
    return s
