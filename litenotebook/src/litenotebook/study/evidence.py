"""Collect a bounded, diverse evidence pool from the material a user selected.

Only the selected sources are ever read: every chunk query is restricted to
those source ids (and to this notebook and ready sources). Notes and chat
history are included only when their toggles are on.

* No focus: passages are spread evenly through each selected source and
  interleaved across sources. No embedding model is needed, so MiniLM is not
  started.
* With a focus: FTS5 (lexical) and, when an embedder is supplied, FAISS
  (semantic, MiniLM) candidates are fused with Reciprocal Rank Fusion inside
  the selected sources, near-duplicates are dropped and no page contributes
  more than two passages.

Low-content chunks (very short, or mostly symbols such as tables of contents)
are skipped while enough other text exists.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from litenotebook.retrieval.hybrid import fts_query, rrf_fuse
from litenotebook.study.model import EXCERPT_CHARS, EvidenceItem, Material
from litenotebook.study.validate import strip_labels

if TYPE_CHECKING:
    from litenotebook.config import Settings
    from litenotebook.embed.embedder import TextEmbedder

MIN_TOKENS = 25
MIN_ALPHA = 0.5
MAX_CHUNK_ROWS = 50_000
NOTE_PASSAGE_CHARS = 1200
MIN_NOTE_CHARS = 40
MAX_PER_PAGE = 2
_WORD = re.compile(r"\w+")


class EvidenceError(ValueError):
    """Nothing usable to build a quiz from (message is shown to the user)."""


@dataclass
class Pool:
    items: list[EvidenceItem]  # unlabelled (label == "")
    timings: dict[str, float] = field(default_factory=dict)
    semantic: bool = False  # MiniLM was used


def _alpha_ratio(text: str) -> float:
    letters = sum(ch.isalpha() for ch in text)
    visible = sum(not ch.isspace() for ch in text)
    return letters / visible if visible else 0.0


def selected_sources(
    conn: sqlite3.Connection, notebook_id: int, source_ids: Sequence[int]
) -> list[sqlite3.Row]:
    """The selected sources that exist in this notebook and are ready."""
    if not source_ids:
        return []
    marks = ",".join("?" * len(source_ids))
    return conn.execute(
        f"SELECT id, original_filename FROM sources WHERE notebook_id = ? "
        f"AND status = 'ready' AND id IN ({marks}) ORDER BY id",
        (notebook_id, *source_ids),
    ).fetchall()


def _source_item(row: sqlite3.Row) -> EvidenceItem:
    return EvidenceItem(
        label="",
        kind="source",
        text=row["text"].strip()[:EXCERPT_CHARS],
        title=row["original_filename"],
        source_id=row["source_id"],
        chunk_id=row["id"],
        page_no=row["page_no"],
        chunk_ord=row["ord"],
    )


def _chunks(
    conn: sqlite3.Connection, notebook_id: int, source_ids: Sequence[int]
) -> list[sqlite3.Row]:
    marks = ",".join("?" * len(source_ids))
    return conn.execute(
        f"SELECT c.id, c.source_id, c.page_no, c.ord, c.text, c.n_tokens, "
        f"s.original_filename FROM chunks c JOIN sources s ON s.id = c.source_id "
        f"WHERE c.notebook_id = ? AND s.notebook_id = ? AND s.status = 'ready' "
        f"AND c.source_id IN ({marks}) ORDER BY c.source_id, c.ord LIMIT ?",
        (notebook_id, notebook_id, *source_ids, MAX_CHUNK_ROWS),
    ).fetchall()


def _usable(rows: list[sqlite3.Row], wanted: int) -> list[sqlite3.Row]:
    good = [
        r
        for r in rows
        if r["n_tokens"] >= MIN_TOKENS and _alpha_ratio(r["text"]) >= MIN_ALPHA
    ]
    return good if len(good) >= min(wanted, len(rows)) else rows


def _spread(rows: list[sqlite3.Row], wanted: int) -> list[sqlite3.Row]:
    """Evenly spaced passages from each source, interleaved across sources."""
    by_source: dict[int, list[sqlite3.Row]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], []).append(r)
    total = len(rows)
    picks: dict[int, list[sqlite3.Row]] = {}
    for sid, group in by_source.items():
        share = max(1, round(wanted * len(group) / total)) if total else 0
        share = min(share, len(group))
        step = len(group) / share
        picks[sid] = [group[int(i * step + step / 2)] for i in range(share)]
    out: list[sqlite3.Row] = []
    depth = 0
    while len(out) < wanted and any(depth < len(p) for p in picks.values()):
        for sid in sorted(picks):
            if depth < len(picks[sid]):
                out.append(picks[sid][depth])
        depth += 1
    return out[:wanted]


def _focused(
    conn: sqlite3.Connection,
    settings: Settings,
    notebook_id: int,
    rows: list[sqlite3.Row],
    focus: str,
    wanted: int,
    embedder: TextEmbedder | None,
    timings: dict[str, float],
) -> list[sqlite3.Row]:
    by_id = {r["id"]: r for r in rows}
    allowed = set(by_id)
    lexical: list[int] = []
    match = fts_query(focus)
    if match is not None:
        started = time.perf_counter()
        marks = ",".join("?" * len({r["source_id"] for r in rows}))
        try:
            lexical = [
                int(r[0])
                for r in conn.execute(
                    f"SELECT c.id FROM chunks_fts JOIN chunks c ON c.id = "
                    f"chunks_fts.rowid WHERE chunks_fts MATCH ? AND c.notebook_id = ? "
                    f"AND c.source_id IN ({marks}) ORDER BY bm25(chunks_fts) LIMIT ?",
                    (
                        match,
                        notebook_id,
                        *sorted({r["source_id"] for r in rows}),
                        settings.retrieval_candidates,
                    ),
                )
                if int(r[0]) in allowed
            ]
        except sqlite3.OperationalError:
            lexical = []
        timings["lexical"] = round(time.perf_counter() - started, 4)
    semantic: list[int] = []
    if embedder is not None:
        from litenotebook.index.faiss_store import FaissStore

        started = time.perf_counter()
        vector = embedder.encode([focus])[0]
        store = FaissStore(conn, settings)
        # The index covers the whole notebook: search all of it, then keep only
        # the selected sources' chunks.
        hits = store.search(notebook_id, vector, max(len(by_id) * 4, 200))
        semantic = [cid for cid, score in hits if cid in allowed and score > 0.15]
        semantic = semantic[: settings.retrieval_candidates]
        timings["semantic"] = round(time.perf_counter() - started, 4)
    fused = rrf_fuse([semantic, lexical], k=settings.rrf_k)
    ranked = [
        by_id[cid] for cid, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    out: list[sqlite3.Row] = []
    seen_words: list[frozenset[str]] = []
    per_page: dict[tuple[int, int], int] = {}
    for r in ranked:
        page = (r["source_id"], r["page_no"])
        if per_page.get(page, 0) >= MAX_PER_PAGE:
            continue
        words = frozenset(w.lower() for w in _WORD.findall(r["text"]))
        if not words or any(
            len(words & s) / len(words | s) > settings.dedup_jaccard_threshold
            for s in seen_words
        ):
            continue
        out.append(r)
        seen_words.append(words)
        per_page[page] = per_page.get(page, 0) + 1
        if len(out) >= wanted:
            break
    return out


def _note_items(
    conn: sqlite3.Connection, notebook_id: int, focus: str, wanted: int
) -> list[EvidenceItem]:
    rows: list[sqlite3.Row] = []
    match = fts_query(focus) if focus else None
    if match is not None:
        try:
            rows = conn.execute(
                "SELECT n.id, n.title, n.content FROM notes_fts JOIN notes n "
                "ON n.id = notes_fts.rowid WHERE notes_fts MATCH ? AND "
                "n.notebook_id = ? ORDER BY bm25(notes_fts) LIMIT 20",
                (match, notebook_id),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    if not rows:
        rows = conn.execute(
            "SELECT id, title, content FROM notes WHERE notebook_id = ? "
            "ORDER BY updated_at DESC, id DESC LIMIT 20",
            (notebook_id,),
        ).fetchall()
    out: list[EvidenceItem] = []
    for r in rows:
        text = (r["content"] or "").strip()
        if len(text) < MIN_NOTE_CHARS:
            continue
        for start in range(0, len(text), NOTE_PASSAGE_CHARS):
            passage = text[start : start + NOTE_PASSAGE_CHARS].strip()
            if len(passage) >= MIN_NOTE_CHARS:
                out.append(
                    EvidenceItem(
                        label="",
                        kind="note",
                        text=passage[:EXCERPT_CHARS],
                        title=r["title"],
                        note_id=r["id"],
                    )
                )
            if len(out) >= wanted:
                return out
    return out


def _chat_items(
    conn: sqlite3.Connection, notebook_id: int, focus: str, wanted: int
) -> list[EvidenceItem]:
    """Completed, model-written answers with their questions."""
    base = (
        "SELECT a.id, a.content, a.conversation_id, q.content AS question, "
        "c.title FROM messages a JOIN messages q ON q.id = a.reply_to "
        "JOIN conversations c ON c.id = a.conversation_id "
        "WHERE c.notebook_id = ? AND a.role = 'assistant' AND a.status = 'complete' "
        "AND a.model IS NOT NULL"
    )
    rows: list[sqlite3.Row] = []
    match = fts_query(focus) if focus else None
    if match is not None:
        try:
            rows = conn.execute(
                base + " AND a.id IN (SELECT rowid FROM messages_fts WHERE "
                "messages_fts MATCH ?) ORDER BY a.id DESC LIMIT ?",
                (notebook_id, match, wanted),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
    if not rows:
        rows = conn.execute(
            base + " ORDER BY a.id DESC LIMIT ?", (notebook_id, wanted)
        ).fetchall()
    out = []
    for r in rows:
        text = (
            f"Question: {' '.join(r['question'].split())}\n"
            f"Answer: {strip_labels(' '.join(r['content'].split()))}"
        )
        out.append(
            EvidenceItem(
                label="",
                kind="chat",
                text=text[:EXCERPT_CHARS],
                title=r["title"],
                message_id=r["id"],
                conversation_id=r["conversation_id"],
            )
        )
    return out


def gather(
    conn: sqlite3.Connection,
    settings: Settings,
    notebook_id: int,
    material: Material,
    wanted: int,
    embedder: TextEmbedder | None = None,
) -> Pool:
    """Up to ``wanted`` passages from the selection. ``embedder`` is used only
    with a focus; pass None to stay lexical."""
    timings: dict[str, float] = {}
    started = time.perf_counter()
    sources = selected_sources(conn, notebook_id, material.source_ids)
    if not sources and not (material.include_notes or material.include_chat):
        raise EvidenceError("Select at least one ready source.")
    source_items: list[EvidenceItem] = []
    focus = " ".join(material.focus.split())
    if sources:
        rows = _usable(_chunks(conn, notebook_id, [r["id"] for r in sources]), wanted)
        if focus:
            picked = _focused(
                conn, settings, notebook_id, rows, focus, wanted, embedder, timings
            )
        else:
            picked = _spread(rows, wanted)
        source_items = [_source_item(r) for r in picked]
    extra_budget = max(2, wanted // 3)
    note_items = (
        _note_items(conn, notebook_id, focus, extra_budget)
        if material.include_notes
        else []
    )
    chat_items = (
        _chat_items(conn, notebook_id, focus, extra_budget)
        if material.include_chat
        else []
    )
    items = source_items + note_items + chat_items
    timings["retrieval"] = round(time.perf_counter() - started, 4)
    if not items:
        if focus:
            raise EvidenceError(
                f"Nothing in the selected material matches “{focus[:80]}”. "
                "Try a different focus, or leave it empty."
            )
        raise EvidenceError("The selected material has no usable text for a quiz.")
    return Pool(
        items=items, timings=timings, semantic=embedder is not None and bool(focus)
    )
