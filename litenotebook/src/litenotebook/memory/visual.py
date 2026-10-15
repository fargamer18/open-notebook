"""The knowledge-graph view: Global and Local graphs over existing data.

A read-only *view* over graph memory (``memory_nodes`` / ``memory_edges``)
plus the notebook's sources, annotations and quizzes. Nothing here writes to
the database, and nothing calls MiniLM, an LLM or speech.

Node types shown can be chosen. A hidden type does not cut the graph apart:
its nodes are *contracted* into the visible object that owns them, and every
stored edge is re-attached to those owners with its original relation:

* question, answer, correction → their conversation
* source evidence → its source
* hidden notes, concepts, annotations and quizzes are left out

So "conversation —mentions→ concept (via question)" is the stored edge
"question —mentions→ concept" seen from the conversation. Each view edge
carries how many stored edges it stands for (``weight``) and which hidden
types it passed through (``via``). No edge is invented: the only edges that
are not ``memory_edges`` rows are the stored references between records
(evidence → its source, annotation → its source, note → linked annotation,
quiz → the concepts its questions test and the sources its evidence cites).

Large graphs are bounded deterministically (degree, then type, label, id)
and the response says how much was left out.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# --- vocabulary -------------------------------------------------------------------

DEFAULT_TYPES = ("concept", "note", "source", "conversation", "correction")
OPTIONAL_TYPES = ("question", "answer", "evidence", "annotation", "quiz")
ALL_TYPES = DEFAULT_TYPES + OPTIONAL_TYPES
INSPECTOR_TYPES = frozenset(DEFAULT_TYPES) | {"evidence", "annotation", "quiz"}
TYPE_ORDER = {t: i for i, t in enumerate(ALL_TYPES)}
TYPE_LABELS = {
    "concept": "Concept",
    "note": "Note",
    "source": "Source",
    "conversation": "Conversation",
    "correction": "Correction",
    "question": "Question",
    "answer": "Answer",
    "evidence": "Evidence",
    "annotation": "Annotation",
    "quiz": "Quiz",
}
_MEMORY_TYPE = {"source_evidence": "evidence"}  # memory node_type → view type
RELATION_LABELS = {
    "contains": "contains",
    "answered_by": "answered by",
    "followed_by": "followed by",
    "mentions": "mentions",
    "supported_by": "supported by",
    "corrects": "corrects",
    "note_mentions": "mentions",
    "note_supported_by": "supported by",
    "derived_from": "passage of",
    "annotates": "annotates",
    "linked_to": "linked to",
    "tests": "tests",
    "references": "references",
}

# Bounds. Defaults were chosen from measurements (docs/graph.md): a 400-node
# view lays out in well under a second in the browser; requests may ask for
# up to the hard caps.
GLOBAL_NODES = 400
GLOBAL_EDGES = 1600
MAX_NODES = 1500
MAX_EDGES = 6000
LOCAL_NODES = 150
LOCAL_EDGES = 600
MAX_DEPTH = 3
LOAD_LIMIT = 50_000  # memory nodes read per request (never the whole DB blindly)
# A concept mentioned by a single object adds a leaf, not structure. Once a
# view has more than this many concepts, "auto" shows only concepts shared by
# at least two objects (the response says how many were hidden).
CONCEPT_AUTO_AT = 40
LABEL_CHARS = 80
SNIPPET_CHARS = 240
NEIGHBOURS_PER_TYPE = 8


def _clip(text: str | None, limit: int = LABEL_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def parse_types(raw: str | None, default: tuple[str, ...] = DEFAULT_TYPES) -> set[str]:
    """Only known type names; anything else is ignored."""
    if raw is None:
        return set(default)
    wanted = {t.strip() for t in raw.split(",")}
    return {t for t in wanted if t in TYPE_ORDER} or set(default)


# --- the view graph ---------------------------------------------------------------


@dataclass
class VNode:
    id: str
    type: str
    label: str
    ref: dict[str, int] = field(default_factory=dict)  # record ids (navigation)
    flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class View:
    nodes: dict[str, VNode]
    edges: dict[tuple[str, str, str], list[Any]]  # (from, to, rel) → [weight, via]
    loaded_all: bool = True

    def adjacency(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {n: set() for n in self.nodes}
        for a, b, _ in self.edges:
            adj[a].add(b)
            adj[b].add(a)
        return adj


def _study_status(streak: int) -> str:
    """The Study tab's labels — from actual attempts, never "mastered"."""
    if streak < 0:
        return "needs_review"
    if streak >= 2:
        return "strong_recently"
    return "seen"


def build_view(conn: sqlite3.Connection, notebook_id: int, types: set[str]) -> View:
    """Nodes of the chosen types and contracted edges between them."""
    nodes: dict[str, VNode] = {}
    edges: dict[tuple[str, str, str], list[Any]] = {}

    def add_edge(a: str | None, b: str | None, rel: str, via: set[str]) -> None:
        if a is None or b is None or a == b or a not in nodes or b not in nodes:
            return
        entry = edges.setdefault((a, b, rel), [0, set()])
        entry[0] += 1
        entry[1] |= via

    # Sources (always known, so evidence can be contracted into them)
    sources: dict[int, sqlite3.Row] = {
        r["id"]: r
        for r in conn.execute(
            "SELECT id, title, page_count, status FROM sources WHERE notebook_id = ?",
            (notebook_id,),
        )
    }
    if "source" in types:
        for sid, r in sources.items():
            flags = [] if r["status"] == "ready" else [f"status_{r['status']}"]
            nodes[f"s:{sid}"] = VNode(
                f"s:{sid}",
                "source",
                _clip(r["title"]),
                {"source": sid},
                flags,
                {"pages": r["page_count"]},
            )

    rows = conn.execute(
        "SELECT id, key, node_type, label, conversation_id, message_id, source_id, "
        "chunk_id, note_id FROM memory_nodes WHERE notebook_id = ? ORDER BY id LIMIT ?",
        (notebook_id, LOAD_LIMIT + 1),
    ).fetchall()
    loaded_all = len(rows) <= LOAD_LIMIT
    rows = rows[:LOAD_LIMIT]
    by_key = {r["key"]: r["id"] for r in rows}
    mtype = {r["id"]: _MEMORY_TYPE.get(r["node_type"], r["node_type"]) for r in rows}
    outcomes = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT message_id, outcome FROM memory_outcomes WHERE notebook_id = ?",
            (notebook_id,),
        )
    }
    study = {
        r[0]: (r[1], r[2], r[3])
        for r in conn.execute(
            "SELECT concept_key, streak, times_seen, last_seen FROM "
            "study_concept_stats WHERE notebook_id = ?",
            (notebook_id,),
        )
    }

    def conversation_node(cid: int | None) -> str | None:
        node = by_key.get(f"conversation:{cid}") if cid is not None else None
        return f"m:{node}" if node is not None else None

    rep: dict[int, str | None] = {}
    for r in rows:
        t = mtype[r["id"]]
        if t in types:
            rep[r["id"]] = f"m:{r['id']}"
            ref = {
                k: r[k]
                for k in ("conversation_id", "message_id", "source_id", "chunk_id")
                if r[k] is not None
            }
            if r["note_id"] is not None:
                ref["note_id"] = r["note_id"]
            ref["memory"] = r["id"]
            node = VNode(f"m:{r['id']}", t, _clip(r["label"]), ref)
            if t == "answer" and r["message_id"] in outcomes:
                node.flags.append(outcomes[r["message_id"]])
                if outcomes[r["message_id"]] == "corrected":
                    node.flags.append("superseded")
            if t == "evidence":
                node.meta["citation"] = r["key"].startswith("evidence:")
                if r["key"].startswith("evidence:"):
                    node.ref["citation_id"] = int(r["key"].split(":", 1)[1])
                if r["source_id"] not in sources:
                    node.flags.append("source_deleted")
            if t == "concept":
                key = r["key"].split(":", 1)[1]
                if key in study:
                    streak, seen, last = study[key]
                    node.flags.append(_study_status(streak))
                    node.meta["study"] = {"times_seen": seen, "last_seen": last}
            nodes[node.id] = node
            continue
        owner: str | None = None
        if t in ("question", "answer", "correction") and "conversation" in types:
            owner = conversation_node(r["conversation_id"])
        elif t == "evidence" and "source" in types and r["source_id"] in sources:
            owner = f"s:{r['source_id']}"
        rep[r["id"]] = owner

    # Conversation outcome summary (answers may be hidden)
    if "conversation" in types:
        for r in conn.execute(
            "SELECT m.conversation_id, o.outcome, count(*) FROM memory_outcomes o "
            "JOIN messages m ON m.id = o.message_id WHERE o.notebook_id = ? "
            "GROUP BY 1, 2",
            (notebook_id,),
        ):
            node_id = conversation_node(r[0])
            if node_id in nodes:
                nodes[node_id].meta.setdefault("outcomes", {})[r[1]] = r[2]
                if r[1] == "corrected" and "has_correction" not in nodes[node_id].flags:
                    nodes[node_id].flags.append("has_correction")

    for e in conn.execute(
        "SELECT from_node_id, to_node_id, relation FROM memory_edges "
        "WHERE notebook_id = ?",
        (notebook_id,),
    ):
        a, b = rep.get(e[0]), rep.get(e[1])
        via = {mtype[x] for x in (e[0], e[1]) if x in mtype and mtype[x] not in types}
        add_edge(a, b, e[2], via)

    # Stored references between records
    if "evidence" in types and "source" in types:
        for r in rows:
            if mtype[r["id"]] == "evidence" and r["source_id"] in sources:
                add_edge(f"m:{r['id']}", f"s:{r['source_id']}", "derived_from", set())
    if "annotation" in types:
        for r in conn.execute(
            "SELECT id, source_id, page_no, annotation_type, selected_text, comment "
            "FROM annotations WHERE notebook_id = ?",
            (notebook_id,),
        ):
            text = (
                r["selected_text"] or r["comment"] or f"Comment on page {r['page_no']}"
            )
            nodes[f"a:{r['id']}"] = VNode(
                f"a:{r['id']}",
                "annotation",
                _clip(text),
                {"annotation": r["id"], "source_id": r["source_id"]},
                [],
                {"page": r["page_no"], "kind": r["annotation_type"]},
            )
            add_edge(f"a:{r['id']}", f"s:{r['source_id']}", "annotates", set())
        if "note" in types:
            for r in conn.execute(
                "SELECT l.note_id, l.annotation_id FROM note_links l "
                "JOIN notes n ON n.id = l.note_id WHERE n.notebook_id = ? "
                "AND l.link_type = 'annotation' AND l.annotation_id IS NOT NULL",
                (notebook_id,),
            ):
                note = by_key.get(f"note:{r[0]}")
                if note is not None:
                    add_edge(f"m:{note}", f"a:{r[1]}", "linked_to", set())
    if "quiz" in types:
        _add_quizzes(conn, notebook_id, nodes, by_key, add_edge)
    return View(nodes, edges, loaded_all)


def _add_quizzes(
    conn: sqlite3.Connection,
    notebook_id: int,
    nodes: dict[str, VNode],
    by_key: dict[str, int],
    add_edge: Any,
) -> None:
    for q in conn.execute(
        "SELECT id, title FROM quizzes WHERE notebook_id = ? AND status = 'ready'",
        (notebook_id,),
    ):
        latest = conn.execute(
            "SELECT id, score, total FROM quiz_attempts WHERE quiz_id = ? "
            "AND status = 'completed' ORDER BY completed_at DESC, id DESC LIMIT 1",
            (q["id"],),
        ).fetchone()
        ref = {"quiz": q["id"]}
        meta: dict[str, Any] = {}
        if latest is not None:
            ref["attempt"] = latest["id"]
            meta["latest_score"] = f"{latest['score']}/{latest['total']}"
        node_id = f"z:{q['id']}"
        nodes[node_id] = VNode(node_id, "quiz", _clip(q["title"]), ref, [], meta)
        keys: set[str] = set()
        for (raw,) in conn.execute(
            "SELECT concepts FROM quiz_questions WHERE quiz_id = ?", (q["id"],)
        ):
            try:
                keys |= {c.get("key", "") for c in json.loads(raw or "[]")}
            except (ValueError, AttributeError):
                continue
        for key in sorted(keys):
            concept = by_key.get(f"concept:{key}")
            if concept is not None:
                add_edge(node_id, f"m:{concept}", "tests", set())
        for (sid,) in conn.execute(
            "SELECT DISTINCT e.source_id FROM quiz_evidence e JOIN quiz_questions q "
            "ON q.id = e.question_id WHERE q.quiz_id = ? AND e.source_id IS NOT NULL",
            (q["id"],),
        ):
            add_edge(node_id, f"s:{sid}", "references", set())


# --- payloads -----------------------------------------------------------------------


def _degrees(view: View) -> dict[str, int]:
    deg = dict.fromkeys(view.nodes, 0)
    for a, b, _ in view.edges:
        deg[a] += 1
        deg[b] += 1
    return deg


def _rank(view: View, deg: dict[str, int]) -> list[str]:
    return sorted(
        view.nodes,
        key=lambda n: (
            -deg[n],
            TYPE_ORDER[view.nodes[n].type],
            view.nodes[n].label.casefold(),
            n,
        ),
    )


def _payload(
    view: View,
    keep: list[str],
    deg: dict[str, int],
    max_edges: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    kept = set(keep)
    edges = [(k, v) for k, v in view.edges.items() if k[0] in kept and k[1] in kept]
    edges.sort(key=lambda kv: (-kv[1][0], kv[0]))
    edges_total = len(edges)
    edges = edges[:max_edges]
    return {
        "nodes": [
            {
                "id": n,
                "type": view.nodes[n].type,
                "label": view.nodes[n].label,
                "degree": deg[n],
                "flags": view.nodes[n].flags,
                "meta": view.nodes[n].meta,
            }
            for n in keep
        ],
        "edges": [
            {
                "source": a,
                "target": b,
                "type": rel,
                "label": RELATION_LABELS.get(rel, rel),
                "weight": w,
                "via": sorted(via),
            }
            for (a, b, rel), (w, via) in edges
        ],
        "edges_total": edges_total,
        "edges_truncated": edges_total > len(edges),
        "complete_load": view.loaded_all,
        **extra,
    }


def global_graph(
    conn: sqlite3.Connection,
    notebook_id: int,
    types: set[str] | None = None,
    *,
    min_degree: int = 0,
    hide_orphans: bool = False,
    concept_min: int | None = None,
    max_nodes: int = GLOBAL_NODES,
    max_edges: int = GLOBAL_EDGES,
) -> dict[str, Any]:
    """``concept_min`` = minimum connections for concepts (None = auto)."""
    chosen = set(types or DEFAULT_TYPES) & set(ALL_TYPES)
    max_nodes = max(1, min(MAX_NODES, int(max_nodes)))
    max_edges = max(1, min(MAX_EDGES, int(max_edges)))
    view = build_view(conn, notebook_id, chosen)
    deg = _degrees(view)
    floor = max(int(min_degree), 1 if hide_orphans else 0)
    concepts = [n for n, v in view.nodes.items() if v.type == "concept"]
    if concept_min is None:
        concept_floor = 2 if len(concepts) > CONCEPT_AUTO_AT else 0
    else:
        concept_floor = max(0, min(50, int(concept_min)))
    concepts_hidden = sum(1 for n in concepts if floor <= deg[n] < concept_floor)
    eligible = [
        n
        for n in _rank(view, deg)
        if deg[n] >= floor
        and (view.nodes[n].type != "concept" or deg[n] >= concept_floor)
    ]
    keep = eligible[:max_nodes]
    return _payload(
        view,
        keep,
        deg,
        max_edges,
        {
            "mode": "global",
            "types": sorted(chosen, key=TYPE_ORDER.__getitem__),
            "nodes_total": len(eligible),
            "nodes_in_notebook": len(view.nodes),
            "concept_min": concept_floor,
            "concept_min_auto": concept_min is None,
            "concepts_hidden": concepts_hidden,
            "truncated": len(eligible) > len(keep),
            "limits": {"nodes": max_nodes, "edges": max_edges},
        },
    )


def local_graph(
    conn: sqlite3.Connection,
    notebook_id: int,
    center: str,
    depth: int = 1,
    types: set[str] | None = None,
    *,
    max_nodes: int = LOCAL_NODES,
) -> dict[str, Any] | None:
    """The centre and its neighbourhood up to ``depth`` (1–3), breadth first
    in a deterministic order, at most ``max_nodes`` nodes. None if the centre
    is not in this notebook."""
    depth = max(1, min(MAX_DEPTH, int(depth)))
    max_nodes = max(1, min(MAX_NODES, int(max_nodes)))
    chosen = set(types or DEFAULT_TYPES) & set(ALL_TYPES)
    kind = node_kind(conn, notebook_id, center)
    if kind is None:
        return None
    chosen.add(kind)  # the centre is always shown
    view = build_view(conn, notebook_id, chosen)
    if center not in view.nodes:
        return None
    adj = view.adjacency()
    deg = _degrees(view)
    level = {center: 0}
    frontier = [center]
    truncated = False
    for d in range(1, depth + 1):
        found = sorted(
            {n for f in frontier for n in adj[f] if n not in level},
            key=lambda n: (
                TYPE_ORDER[view.nodes[n].type],
                -deg[n],
                view.nodes[n].label.casefold(),
                n,
            ),
        )
        frontier = []
        for n in found:
            if len(level) >= max_nodes:
                truncated = True
                break
            level[n] = d
            frontier.append(n)
        if not frontier:
            break
    reachable = _reachable_count(adj, center, depth)
    payload = _payload(
        view,
        list(level),
        deg,
        LOCAL_EDGES,
        {
            "mode": "local",
            "center": center,
            "depth": depth,
            "types": sorted(chosen, key=TYPE_ORDER.__getitem__),
            "nodes_total": reachable,
            "truncated": truncated,
            "limits": {"nodes": max_nodes, "edges": LOCAL_EDGES, "depth": MAX_DEPTH},
        },
    )
    for node in payload["nodes"]:
        node["depth"] = level[node["id"]]
    return payload


def _reachable_count(adj: dict[str, set[str]], center: str, depth: int) -> int:
    seen = {center}
    queue = deque([(center, 0)])
    while queue:
        node, d = queue.popleft()
        if d == depth:
            continue
        for n in adj[node]:
            if n not in seen:
                seen.add(n)
                queue.append((n, d + 1))
    return len(seen)


# --- single nodes -------------------------------------------------------------------


def node_kind(conn: sqlite3.Connection, notebook_id: int, node_id: str) -> str | None:
    """The view type of a node id in this notebook (ownership check)."""
    prefix, _, raw = node_id.partition(":")
    if not raw.isdigit() or len(raw) > 18:
        return None
    ident = int(raw)
    if prefix == "m":
        row = conn.execute(
            "SELECT node_type FROM memory_nodes WHERE id = ? AND notebook_id = ?",
            (ident, notebook_id),
        ).fetchone()
        return _MEMORY_TYPE.get(row[0], row[0]) if row else None
    table = {
        "s": ("sources", "source"),
        "a": ("annotations", "annotation"),
        "z": ("quizzes", "quiz"),
    }.get(prefix)
    if table is None:
        return None
    row = conn.execute(
        f"SELECT 1 FROM {table[0]} WHERE id = ? AND notebook_id = ?",
        (ident, notebook_id),
    ).fetchone()
    return table[1] if row else None


def actions(notebook_id: int, node: VNode) -> list[dict[str, str]]:
    """Links into the real product for a node (first = primary)."""
    base = f"/notebooks/{notebook_id}"
    r = node.ref
    out: list[dict[str, str]] = []
    chat = (
        f"{base}/chat/{r['conversation_id']}"
        + (f"#msg-{r['message_id']}" if "message_id" in r else "")
        if "conversation_id" in r
        else None
    )
    t = node.type
    if t == "note" and "note_id" in r:
        out.append({"label": "Open note", "href": f"{base}/notes/{r['note_id']}"})
    elif t == "conversation" and chat:
        out.append(
            {"label": "Open chat", "href": f"{base}/chat/{r['conversation_id']}"}
        )
    elif t in ("question", "answer", "correction") and chat:
        out.append({"label": "Open in chat", "href": chat})
    elif t == "source":
        out.append({"label": "Read PDF", "href": f"{base}/sources/{r['source']}/read"})
        out.append({"label": "Open source", "href": f"{base}/sources/{r['source']}"})
    elif t == "evidence" and "source_deleted" not in node.flags and "source_id" in r:
        reader = f"{base}/sources/{r['source_id']}/read"
        if "citation_id" in r:
            out.append(
                {
                    "label": "Open passage in PDF",
                    "href": f"{reader}?citation={r['citation_id']}",
                }
            )
        elif "chunk_id" in r:
            out.append(
                {
                    "label": "Open passage in PDF",
                    "href": f"{reader}?chunk={r['chunk_id']}",
                }
            )
        if chat:
            out.append({"label": "Open the answer in chat", "href": chat})
    elif t == "annotation":
        out.append(
            {
                "label": "Open in PDF",
                "href": (
                    f"{base}/sources/{r['source_id']}/read?annotation={r['annotation']}"
                ),
            }
        )
    elif t == "quiz":
        out.append({"label": "Open quiz", "href": f"{base}/study/quizzes/{r['quiz']}"})
        if "attempt" in r:
            out.append(
                {
                    "label": "Review latest attempt",
                    "href": f"{base}/study/attempts/{r['attempt']}/review",
                }
            )
    if "memory" in r:
        out.append(
            {"label": "Memory details", "href": f"{base}/memory/nodes/{r['memory']}"}
        )
    return out


def _snippet(conn: sqlite3.Connection, notebook_id: int, node: VNode) -> str:
    """A short excerpt for the inspector (never a whole document)."""
    r = node.ref
    if "memory" in r and node.type != "concept":
        row = conn.execute(
            "SELECT content FROM memory_nodes WHERE id = ? AND notebook_id = ?",
            (r["memory"], notebook_id),
        ).fetchone()
        return _clip(row[0] if row else "", SNIPPET_CHARS)
    if node.type == "annotation":
        row = conn.execute(
            "SELECT comment FROM annotations WHERE id = ? AND notebook_id = ?",
            (r["annotation"], notebook_id),
        ).fetchone()
        return _clip(row[0] if row else "", SNIPPET_CHARS)
    return ""


def node_detail(
    conn: sqlite3.Connection, notebook_id: int, node_id: str
) -> dict[str, Any] | None:
    """Inspector data: the node, a short snippet, links into the product and
    its connections grouped by type (a few of each). Connections use the same
    contracted view as the graph — the default types plus evidence,
    annotations and quizzes — so a conversation lists its cited passages and
    a concept its notes and conversations; questions and answers are folded
    into their conversation unless the node is one of them."""
    kind = node_kind(conn, notebook_id, node_id)
    if kind is None:
        return None
    view = build_view(conn, notebook_id, set(INSPECTOR_TYPES) | {kind})
    node = view.nodes.get(node_id)
    if node is None:
        return None
    groups: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for a, b, rel in sorted(view.edges):
        if node_id not in (a, b):
            continue
        other = b if a == node_id else a
        if other in seen:
            continue
        seen.add(other)
        o = view.nodes[other]
        counts[o.type] = counts.get(o.type, 0) + 1
        bucket = groups.setdefault(o.type, [])
        if len(bucket) < NEIGHBOURS_PER_TYPE:
            bucket.append(
                {
                    "id": other,
                    "type": o.type,
                    "label": o.label,
                    "relation": RELATION_LABELS.get(rel, rel),
                    "direction": "out" if a == node_id else "in",
                    "flags": o.flags,
                }
            )
    for bucket in groups.values():
        bucket.sort(key=lambda n: (n["label"].casefold(), n["id"]))
    return {
        "id": node.id,
        "type": node.type,
        "type_label": TYPE_LABELS[node.type],
        "label": node.label,
        "flags": node.flags,
        "meta": node.meta,
        "snippet": _snippet(conn, notebook_id, node),
        "actions": actions(notebook_id, node),
        "connections": {
            t: {"count": counts[t], "items": groups[t]}
            for t in sorted(groups, key=TYPE_ORDER.__getitem__)
        },
    }


def search(
    conn: sqlite3.Connection,
    notebook_id: int,
    query: str,
    types: set[str] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Case-insensitive substring match on labels (no MiniLM, no LLM).
    Prefix matches first, then better-connected nodes."""
    q = " ".join(query.split()).casefold()[:200]
    if not q:
        return []
    view = build_view(conn, notebook_id, set(types or ALL_TYPES) & set(ALL_TYPES))
    deg = _degrees(view)
    hits = [n for n in view.nodes.values() if q in n.label.casefold()]
    hits.sort(
        key=lambda n: (
            not n.label.casefold().startswith(q),
            -deg[n.id],
            TYPE_ORDER[n.type],
            n.label.casefold(),
            n.id,
        )
    )
    return [
        {"id": n.id, "type": n.type, "label": n.label, "degree": deg[n.id]}
        for n in hits[: max(1, min(50, limit))]
    ]
