"""Bounded local subgraphs of graph memory for the Memory Graph explorer.

Never the whole notebook graph: a centre node plus neighbours up to depth 2,
at most MAX_NODES nodes and MAX_EDGES edges, expanded level by level with
indexed queries in a deterministic order (type, label, id). SQLite only; no
MiniLM, no LLM.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from litenotebook.memory.search import fts_match

MAX_NODES = 50
MAX_EDGES = 100
MAX_DEPTH = 2
MAX_NEIGHBOUR_ROWS = 400  # per level, before capping
LABEL_CHARS = 60
TYPE_ORDER = {
    "concept": 0,
    "note": 1,
    "question": 2,
    "answer": 3,
    "correction": 4,
    "source_evidence": 5,
    "conversation": 6,
}
FILTERABLE = tuple(TYPE_ORDER)


def _clip(text: str, limit: int = LABEL_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def default_center(conn: sqlite3.Connection, notebook_id: int) -> int | None:
    """The best-connected concept (else the newest node)."""
    row = conn.execute(
        "SELECT n.id FROM memory_nodes n LEFT JOIN memory_edges e "
        "ON e.to_node_id = n.id OR e.from_node_id = n.id "
        "WHERE n.notebook_id = ? AND n.node_type = 'concept' "
        "GROUP BY n.id ORDER BY count(e.id) DESC, n.id LIMIT 1",
        (notebook_id,),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT id FROM memory_nodes WHERE notebook_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (notebook_id,),
        ).fetchone()
    return int(row[0]) if row else None


def subgraph(
    conn: sqlite3.Connection,
    notebook_id: int,
    center_id: int,
    depth: int = 1,
    types: set[str] | None = None,
) -> dict[str, Any] | None:
    """Local neighbourhood of ``center_id`` in this notebook, or None."""
    depth = max(1, min(MAX_DEPTH, int(depth)))
    allowed = {t for t in (types or set()) if t in TYPE_ORDER} or set(TYPE_ORDER)
    center = conn.execute(
        "SELECT id FROM memory_nodes WHERE id = ? AND notebook_id = ?",
        (center_id, notebook_id),
    ).fetchone()
    if center is None:
        return None
    level: dict[int, int] = {center_id: 0}
    frontier = [center_id]
    truncated = False
    for d in range(1, depth + 1):
        if not frontier:
            break
        marks = ",".join("?" * len(frontier))
        rows = conn.execute(
            "SELECT n.id, n.node_type, n.label FROM memory_edges e "
            "JOIN memory_nodes n ON n.id = CASE WHEN e.from_node_id IN "
            f"({marks}) THEN e.to_node_id ELSE e.from_node_id END "
            f"WHERE e.notebook_id = ? AND (e.from_node_id IN ({marks}) "
            f"OR e.to_node_id IN ({marks})) AND n.notebook_id = ? LIMIT ?",
            (
                *frontier,
                notebook_id,
                *frontier,
                *frontier,
                notebook_id,
                MAX_NEIGHBOUR_ROWS,
            ),
        ).fetchall()
        if len(rows) >= MAX_NEIGHBOUR_ROWS:
            truncated = True
        found = {
            r["id"]: (TYPE_ORDER.get(r["node_type"], 9), r["label"])
            for r in rows
            if r["id"] not in level and r["node_type"] in allowed
        }
        next_frontier = []
        for node_id in sorted(found, key=lambda i: (*found[i], i)):
            if len(level) >= MAX_NODES:
                truncated = True
                break
            level[node_id] = d
            next_frontier.append(node_id)
        frontier = next_frontier

    ids = list(level)
    marks = ",".join("?" * len(ids))
    nodes = [
        {
            "id": r["id"],
            "type": r["node_type"],
            "label": _clip(r["label"]),
            "provenance": r["provenance"],
            "depth": level[r["id"]],
        }
        for r in conn.execute(
            "SELECT id, node_type, label, provenance FROM memory_nodes "
            f"WHERE notebook_id = ? AND id IN ({marks})",
            (notebook_id, *ids),
        )
    ]
    nodes.sort(
        key=lambda n: (n["depth"], TYPE_ORDER.get(n["type"], 9), n["label"], n["id"])
    )
    edge_rows = conn.execute(
        "SELECT id, from_node_id, to_node_id, relation FROM memory_edges "
        f"WHERE notebook_id = ? AND from_node_id IN ({marks}) "
        f"AND to_node_id IN ({marks}) ORDER BY id LIMIT ?",
        (notebook_id, *ids, *ids, MAX_EDGES + 1),
    ).fetchall()
    if len(edge_rows) > MAX_EDGES:
        truncated = True
        edge_rows = edge_rows[:MAX_EDGES]
    edges = [
        {
            "id": r["id"],
            "from": r["from_node_id"],
            "to": r["to_node_id"],
            "relation": r["relation"],
        }
        for r in edge_rows
    ]
    return {
        "center": center_id,
        "depth": depth,
        "types": sorted(allowed),
        "nodes": nodes,
        "edges": edges,
        "truncated": truncated,
        "limits": {"nodes": MAX_NODES, "edges": MAX_EDGES, "depth": MAX_DEPTH},
    }


def edge_detail(
    conn: sqlite3.Connection, notebook_id: int, edge_id: int
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT e.id, e.relation, e.provenance, e.created_at, "
        "a.id AS from_id, a.label AS from_label, a.node_type AS from_type, "
        "b.id AS to_id, b.label AS to_label, b.node_type AS to_type "
        "FROM memory_edges e JOIN memory_nodes a ON a.id = e.from_node_id "
        "JOIN memory_nodes b ON b.id = e.to_node_id "
        "WHERE e.id = ? AND e.notebook_id = ? AND a.notebook_id = ? "
        "AND b.notebook_id = ?",
        (edge_id, notebook_id, notebook_id, notebook_id),
    ).fetchone()
    return dict(row) if row else None


def search_nodes(
    conn: sqlite3.Connection, notebook_id: int, query: str, limit: int = 12
) -> list[dict[str, Any]]:
    match = fts_match(query[:300])
    if match is None:
        return []
    try:
        rows = conn.execute(
            "SELECT n.id, n.node_type, n.label FROM memory_nodes_fts f "
            "JOIN memory_nodes n ON n.id = f.rowid WHERE memory_nodes_fts MATCH ? "
            "AND n.notebook_id = ? "
            "ORDER BY n.node_type <> 'concept', bm25(memory_nodes_fts, 4.0, 1.0), n.id "
            "LIMIT ?",
            (match, notebook_id, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"id": r["id"], "type": r["node_type"], "label": _clip(r["label"])}
        for r in rows
    ]
