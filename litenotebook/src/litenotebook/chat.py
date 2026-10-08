"""Persistent chat: conversations, messages and citation snapshots.

No heavy imports: rendering conversations and history never loads MiniLM,
FAISS or torch.

Message lifecycle (see migration 0004)::

    send:   user message (complete) + assistant message (generating), one tx
            -> complete | failed | interrupted
    retry:  a failed/interrupted assistant message goes back to generating in
            place (same row, same user message, citations cleared)
    start:  any message left generating by a previous process -> interrupted
    reconcile: a generating message that no live generation owns (e.g. the
            browser vanished before the stream started) -> interrupted

Every lookup is scoped by notebook id: ids from another notebook are "not
found".
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from litenotebook.rag.prompts import Turn

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
MAX_QUESTION_CHARS = 4000
MAX_TITLE_CHARS = 120
AUTO_TITLE_CHARS = 60
INTERRUPTED_MESSAGE = "Generation interrupted. Retry."
RESTART_MESSAGE = "Generation interrupted by an application restart. Retry."


class ChatError(ValueError):
    pass


class ConversationNotFoundError(ChatError):
    pass


class MessageNotFoundError(ChatError):
    pass


class CitationNotFoundError(ChatError):
    pass


class ChatBusyError(ChatError):
    pass


class MemoryRefNotFoundError(ChatError):
    pass


@dataclass(frozen=True)
class Conversation:
    id: int
    notebook_id: int
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


@dataclass(frozen=True)
class CitationRecord:
    id: int
    message_id: int
    label: int
    ordinal: int
    chunk_id: int
    source_id: int
    source_filename: str
    page_no: int
    chunk_ord: int
    chunk_text: str
    chunk_exists: bool  # the cited chunk still exists in its original source
    source_exists: bool


@dataclass(frozen=True)
class MemoryRefRecord:
    """An [M#] used by an answer: an immutable snapshot of the memory shown."""

    id: int
    message_id: int
    label: int
    ordinal: int
    node_key: str
    kind: str
    provenance: str
    conversation_id: int | None
    conversation_title: str
    question: str
    excerpt: str
    outcome: str | None
    correction: str | None
    sources: list[dict[str, Any]]
    remembered_at: str
    conversation_exists: bool

    @property
    def trust(self) -> str:
        if self.kind == "correction":
            return "User correction"
        if self.kind == "question":
            return "User statement"
        return "Source-backed answer" if self.sources else "Uncited assistant answer"


@dataclass(frozen=True)
class RememberedItem:
    """What gets snapshotted for one valid [M#] label."""

    label: int
    node_key: str
    kind: str
    provenance: str
    conversation_id: int | None
    conversation_title: str
    question: str
    excerpt: str
    outcome: str | None
    correction: str | None
    sources: list[dict[str, Any]]
    remembered_at: str


@dataclass(frozen=True)
class Message:
    id: int
    conversation_id: int
    role: str
    content: str
    status: str
    reply_to: int | None
    model: str | None
    error: str | None
    meta: dict[str, Any]
    created_at: str
    completed_at: str | None
    citations: list[CitationRecord] = field(default_factory=list)
    memory_refs: list[MemoryRefRecord] = field(default_factory=list)
    outcome: str | None = None
    correction: str | None = None

    @property
    def rateable(self) -> bool:
        return (
            self.role == "assistant"
            and self.status == "complete"
            and self.model is not None
        )

    def memory_ref(self, label: int) -> MemoryRefRecord | None:
        return next((r for r in self.memory_refs if r.label == label), None)

    @property
    def retryable(self) -> bool:
        return self.role == "assistant" and self.status in ("failed", "interrupted")

    def citation(self, label: int) -> CitationRecord | None:
        return next((c for c in self.citations if c.label == label), None)


@dataclass(frozen=True)
class Exchange:
    conversation: Conversation
    user_message_id: int
    assistant_message_id: int
    question: str
    created_conversation: bool = False


@dataclass(frozen=True)
class CitedChunk:
    """What gets snapshotted for one valid [S#] label."""

    label: int
    chunk_id: int
    source_id: int
    source_filename: str
    page_no: int
    chunk_ord: int
    chunk_text: str


# --- validation & titles ------------------------------------------------------------


def clean_question(question: str) -> str:
    question = (question or "").replace("\r\n", "\n").strip()
    if not question:
        raise ChatError("Type a question first")
    if len(question) > MAX_QUESTION_CHARS:
        raise ChatError(f"Questions are limited to {MAX_QUESTION_CHARS} characters")
    if "\x00" in question:
        raise ChatError("The question contains invalid characters")
    return question


def title_from(question: str) -> str:
    """Deterministic title: the first ~60 characters, cut at a word boundary."""
    text = " ".join(question.split())
    if len(text) <= AUTO_TITLE_CHARS:
        return text or "New chat"
    cut = text[:AUTO_TITLE_CHARS]
    if " " in cut[20:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,.;:-") + "…"


def clean_title(title: str) -> str:
    title = " ".join((title or "").split())
    if not title:
        raise ChatError("The title cannot be empty")
    if len(title) > MAX_TITLE_CHARS:
        raise ChatError(f"Titles are limited to {MAX_TITLE_CHARS} characters")
    return title


# --- conversations ------------------------------------------------------------------

_CONV_SELECT = (
    "SELECT c.*, (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) "
    "AS n_messages FROM conversations c"
)


def _conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=row["id"],
        notebook_id=row["notebook_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=row["n_messages"],
    )


def list_conversations(
    conn: sqlite3.Connection, notebook_id: int
) -> list[Conversation]:
    rows = conn.execute(
        _CONV_SELECT + " WHERE c.notebook_id = ? ORDER BY c.updated_at DESC, c.id DESC",
        (notebook_id,),
    ).fetchall()
    return [_conversation(r) for r in rows]


def get_conversation(
    conn: sqlite3.Connection, notebook_id: int, conversation_id: int
) -> Conversation:
    row = conn.execute(
        _CONV_SELECT + " WHERE c.id = ? AND c.notebook_id = ?",
        (conversation_id, notebook_id),
    ).fetchone()
    if row is None:
        raise ConversationNotFoundError(
            f"Conversation {conversation_id} not found in this notebook"
        )
    return _conversation(row)


def create_conversation(
    conn: sqlite3.Connection, notebook_id: int, title: str
) -> Conversation:
    with conn:
        cur = conn.execute(
            "INSERT INTO conversations (notebook_id, title) VALUES (?, ?)",
            (notebook_id, clean_title(title)),
        )
    assert cur.lastrowid is not None
    return get_conversation(conn, notebook_id, cur.lastrowid)


def rename_conversation(
    conn: sqlite3.Connection, notebook_id: int, conversation_id: int, title: str
) -> Conversation:
    get_conversation(conn, notebook_id, conversation_id)
    title = clean_title(title)
    with conn:
        conn.execute(
            f"UPDATE conversations SET title = ?, updated_at = {NOW_SQL} "
            "WHERE id = ? AND notebook_id = ?",
            (title, conversation_id, notebook_id),
        )
    return get_conversation(conn, notebook_id, conversation_id)


def delete_conversation(
    conn: sqlite3.Connection, notebook_id: int, conversation_id: int
) -> Conversation:
    conversation = get_conversation(conn, notebook_id, conversation_id)
    with conn:
        busy = conn.execute(
            "SELECT 1 FROM messages WHERE conversation_id = ? "
            "AND status = 'generating'",
            (conversation_id,),
        ).fetchone()
        if busy:
            raise ChatBusyError(
                "An answer is being generated in this conversation; stop it first"
            )
        conn.execute(
            "DELETE FROM conversations WHERE id = ? AND notebook_id = ?",
            (conversation_id, notebook_id),
        )
        # its memory nodes cascaded; drop concepts nothing mentions any more
        from litenotebook.memory.graph import prune_orphan_concepts

        prune_orphan_concepts(conn, notebook_id)
    return conversation


# --- messages -----------------------------------------------------------------------

_CITATION_SELECT = (
    "SELECT mc.*, "
    "EXISTS (SELECT 1 FROM chunks ch WHERE ch.id = mc.chunk_id "
    "        AND ch.source_id = mc.source_id) AS chunk_exists, "
    "EXISTS (SELECT 1 FROM sources s WHERE s.id = mc.source_id) AS source_exists "
    "FROM message_citations mc"
)


def _citation(row: sqlite3.Row) -> CitationRecord:
    return CitationRecord(
        id=row["id"],
        message_id=row["message_id"],
        label=row["label"],
        ordinal=row["ordinal"],
        chunk_id=row["chunk_id"],
        source_id=row["source_id"],
        source_filename=row["source_filename"],
        page_no=row["page_no"],
        chunk_ord=row["chunk_ord"],
        chunk_text=row["chunk_text"],
        chunk_exists=bool(row["chunk_exists"]),
        source_exists=bool(row["source_exists"]),
    )


def _message(
    row: sqlite3.Row,
    citations: list[CitationRecord],
    memory_refs: list[MemoryRefRecord] | None = None,
    outcome: tuple[str, str | None] | None = None,
) -> Message:
    try:
        meta: dict[str, Any] = json.loads(row["meta"] or "{}")
    except (TypeError, ValueError):
        meta = {}
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        status=row["status"],
        reply_to=row["reply_to"],
        model=row["model"],
        error=row["error"],
        meta=meta,
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        citations=citations,
        memory_refs=memory_refs or [],
        outcome=outcome[0] if outcome else None,
        correction=outcome[1] if outcome else None,
    )


def _citations_for(
    conn: sqlite3.Connection, message_ids: Sequence[int]
) -> dict[int, list[CitationRecord]]:
    out: dict[int, list[CitationRecord]] = {i: [] for i in message_ids}
    if not message_ids:
        return out
    rows = conn.execute(
        _CITATION_SELECT + " WHERE mc.message_id IN (SELECT value FROM json_each(?)) "
        "ORDER BY mc.message_id, mc.ordinal",
        (json.dumps(list(message_ids)),),
    ).fetchall()
    for row in rows:
        out[row["message_id"]].append(_citation(row))
    return out


_MEMREF_SELECT = (
    "SELECT r.*, EXISTS (SELECT 1 FROM conversations c "
    "WHERE c.id = r.conversation_id) AS conversation_exists "
    "FROM message_memory_refs r"
)


def _memory_ref(row: sqlite3.Row) -> MemoryRefRecord:
    try:
        sources = json.loads(row["sources"] or "[]")
    except (TypeError, ValueError):
        sources = []
    return MemoryRefRecord(
        id=row["id"],
        message_id=row["message_id"],
        label=row["label"],
        ordinal=row["ordinal"],
        node_key=row["node_key"],
        kind=row["kind"],
        provenance=row["provenance"],
        conversation_id=row["conversation_id"],
        conversation_title=row["conversation_title"],
        question=row["question"],
        excerpt=row["excerpt"],
        outcome=row["outcome"],
        correction=row["correction"],
        sources=sources if isinstance(sources, list) else [],
        remembered_at=row["remembered_at"],
        conversation_exists=bool(row["conversation_exists"]),
    )


def _extras_for(
    conn: sqlite3.Connection, message_ids: Sequence[int]
) -> tuple[dict[int, list[MemoryRefRecord]], dict[int, tuple[str, str | None]]]:
    refs: dict[int, list[MemoryRefRecord]] = {i: [] for i in message_ids}
    outcomes: dict[int, tuple[str, str | None]] = {}
    if not message_ids:
        return refs, outcomes
    ids = json.dumps(list(message_ids))
    for row in conn.execute(
        _MEMREF_SELECT + " WHERE r.message_id IN (SELECT value FROM json_each(?)) "
        "ORDER BY r.message_id, r.ordinal",
        (ids,),
    ):
        refs[row["message_id"]].append(_memory_ref(row))
    for row in conn.execute(
        "SELECT message_id, outcome, correction FROM memory_outcomes "
        "WHERE message_id IN (SELECT value FROM json_each(?))",
        (ids,),
    ):
        outcomes[row["message_id"]] = (row["outcome"], row["correction"])
    return refs, outcomes


def list_messages(conn: sqlite3.Connection, conversation_id: int) -> list[Message]:
    rows = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    answer_ids = [r["id"] for r in rows if r["role"] == "assistant"]
    citations = _citations_for(conn, answer_ids)
    refs, outcomes = _extras_for(conn, answer_ids)
    return [
        _message(
            r, citations.get(r["id"], []), refs.get(r["id"]), outcomes.get(r["id"])
        )
        for r in rows
    ]


def get_message(conn: sqlite3.Connection, notebook_id: int, message_id: int) -> Message:
    row = conn.execute(
        "SELECT m.* FROM messages m JOIN conversations c ON c.id = m.conversation_id "
        "WHERE m.id = ? AND c.notebook_id = ?",
        (message_id, notebook_id),
    ).fetchone()
    if row is None:
        raise MessageNotFoundError(f"Message {message_id} not found in this notebook")
    refs, outcomes = _extras_for(conn, [message_id])
    return _message(
        row,
        _citations_for(conn, [message_id])[message_id],
        refs[message_id],
        outcomes.get(message_id),
    )


def get_memory_ref(
    conn: sqlite3.Connection, notebook_id: int, ref_id: int
) -> MemoryRefRecord:
    row = conn.execute(
        _MEMREF_SELECT + " JOIN messages m ON m.id = r.message_id "
        "JOIN conversations c2 ON c2.id = m.conversation_id "
        "WHERE r.id = ? AND c2.notebook_id = ?",
        (ref_id, notebook_id),
    ).fetchone()
    if row is None:
        raise MemoryRefNotFoundError(
            f"Memory reference {ref_id} not found in this notebook"
        )
    from litenotebook.memory.search import refresh_source_state

    ref = _memory_ref(row)
    refresh_source_state(conn, ref.sources)
    return ref


def get_citation(
    conn: sqlite3.Connection, notebook_id: int, citation_id: int
) -> CitationRecord:
    row = conn.execute(
        _CITATION_SELECT + " JOIN messages m ON m.id = mc.message_id "
        "JOIN conversations c ON c.id = m.conversation_id "
        "WHERE mc.id = ? AND c.notebook_id = ?",
        (citation_id, notebook_id),
    ).fetchone()
    if row is None:
        raise CitationNotFoundError(
            f"Citation {citation_id} not found in this notebook"
        )
    return _citation(row)


def start_exchange(
    conn: sqlite3.Connection,
    notebook_id: int,
    conversation_id: int | None,
    question: str,
) -> Exchange:
    """Persist the question and a ``generating`` answer placeholder (one tx).

    With no conversation id, the conversation is created here, so opening
    "New chat" never leaves empty conversations behind.
    """
    question = clean_question(question)
    created = conversation_id is None
    with conn:
        if conversation_id is None:
            cur = conn.execute(
                "INSERT INTO conversations (notebook_id, title) VALUES (?, ?)",
                (notebook_id, title_from(question)),
            )
            assert cur.lastrowid is not None
            conversation_id = cur.lastrowid
        else:
            owned = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND notebook_id = ?",
                (conversation_id, notebook_id),
            ).fetchone()
            if owned is None:
                raise ConversationNotFoundError(
                    f"Conversation {conversation_id} not found in this notebook"
                )
        user = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, status, "
            f"completed_at) VALUES (?, 'user', ?, 'complete', {NOW_SQL})",
            (conversation_id, question),
        )
        assert user.lastrowid is not None
        assistant = conn.execute(
            "INSERT INTO messages (conversation_id, role, status, reply_to) "
            "VALUES (?, 'assistant', 'generating', ?)",
            (conversation_id, user.lastrowid),
        )
        assert assistant.lastrowid is not None
        conn.execute(
            f"UPDATE conversations SET updated_at = {NOW_SQL} WHERE id = ?",
            (conversation_id,),
        )
    return Exchange(
        get_conversation(conn, notebook_id, conversation_id),
        user.lastrowid,
        assistant.lastrowid,
        question,
        created,
    )


def start_retry(
    conn: sqlite3.Connection, notebook_id: int, assistant_message_id: int
) -> Exchange:
    """failed/interrupted → generating, in place. Never adds messages."""
    message = get_message(conn, notebook_id, assistant_message_id)
    if not message.retryable:
        raise ChatBusyError("Only failed or interrupted answers can be retried")
    assert message.reply_to is not None
    question = get_message(conn, notebook_id, message.reply_to).content
    with conn:
        conn.execute(
            "DELETE FROM message_citations WHERE message_id = ?", (message.id,)
        )
        conn.execute(
            "DELETE FROM message_memory_refs WHERE message_id = ?", (message.id,)
        )
        cur = conn.execute(
            "UPDATE messages SET status = 'generating', content = '', error = NULL, "
            "model = NULL, meta = NULL, completed_at = NULL "
            "WHERE id = ? AND status IN ('failed', 'interrupted')",
            (message.id,),
        )
        if cur.rowcount != 1:
            raise ChatBusyError("The answer changed state; reload and try again")
        conn.execute(
            f"UPDATE conversations SET updated_at = {NOW_SQL} WHERE id = ?",
            (message.conversation_id,),
        )
    return Exchange(
        get_conversation(conn, notebook_id, message.conversation_id),
        message.reply_to,
        message.id,
        question,
    )


def history_turns(
    conn: sqlite3.Connection, conversation_id: int, before_message_id: int
) -> list[Turn]:
    """Completed, LLM-answered exchanges before ``before_message_id``."""
    rows = conn.execute(
        "SELECT u.content AS question, a.content AS answer FROM messages a "
        "JOIN messages u ON u.id = a.reply_to "
        "WHERE a.conversation_id = ? AND a.role = 'assistant' "
        "AND a.status = 'complete' AND a.model IS NOT NULL AND u.id < ? "
        "ORDER BY u.id",
        (conversation_id, before_message_id),
    ).fetchall()
    return [Turn(r["question"], r["answer"]) for r in rows]


def complete_message(
    conn: sqlite3.Connection,
    message_id: int,
    content: str,
    *,
    model: str | None,
    citations: Iterable[CitedChunk] = (),
    memory_refs: Iterable[RememberedItem] = (),
    meta: dict[str, Any] | None = None,
) -> bool:
    """generating → complete, with [S#]/[M#] snapshots, in one transaction."""
    with conn:
        cur = conn.execute(
            f"UPDATE messages SET status = 'complete', content = ?, model = ?, "
            f"error = NULL, meta = ?, completed_at = {NOW_SQL} "
            "WHERE id = ? AND status = 'generating'",
            (content, model, json.dumps(meta or {}), message_id),
        )
        if cur.rowcount != 1:
            return False
        conn.executemany(
            "INSERT INTO message_citations (message_id, label, ordinal, chunk_id, "
            "source_id, source_filename, page_no, chunk_ord, chunk_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    message_id,
                    c.label,
                    ordinal,
                    c.chunk_id,
                    c.source_id,
                    c.source_filename,
                    c.page_no,
                    c.chunk_ord,
                    c.chunk_text,
                )
                for ordinal, c in enumerate(citations)
            ],
        )
        conn.executemany(
            "INSERT INTO message_memory_refs (message_id, label, ordinal, node_key, "
            "kind, provenance, conversation_id, conversation_title, question, "
            "excerpt, outcome, correction, sources, remembered_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    message_id,
                    r.label,
                    ordinal,
                    r.node_key,
                    r.kind,
                    r.provenance,
                    r.conversation_id,
                    r.conversation_title,
                    r.question,
                    r.excerpt,
                    r.outcome,
                    r.correction,
                    json.dumps(r.sources),
                    r.remembered_at,
                )
                for ordinal, r in enumerate(memory_refs)
            ],
        )
    return True


def end_message(
    conn: sqlite3.Connection,
    message_id: int,
    status: str,
    error: str,
    *,
    content: str = "",
    model: str | None = None,
    meta: dict[str, Any] | None = None,
) -> bool:
    """generating → failed | interrupted (a partial answer may be kept)."""
    assert status in ("failed", "interrupted")
    with conn:
        cur = conn.execute(
            f"UPDATE messages SET status = ?, error = ?, content = ?, model = ?, "
            f"meta = ?, completed_at = {NOW_SQL} "
            "WHERE id = ? AND status = 'generating'",
            (status, error, content, model, json.dumps(meta or {}), message_id),
        )
    return cur.rowcount == 1


def recover_generating(conn: sqlite3.Connection) -> int:
    """Startup: nothing can be generating in a fresh process."""
    with conn:
        cur = conn.execute(
            f"UPDATE messages SET status = 'interrupted', error = ?, "
            f"completed_at = {NOW_SQL} WHERE status = 'generating'",
            (RESTART_MESSAGE,),
        )
    return cur.rowcount


def reconcile_generating(conn: sqlite3.Connection, active: set[int]) -> int:
    """Interrupt generating messages that no live generation owns."""
    rows = conn.execute(
        "SELECT id FROM messages WHERE status = 'generating'"
    ).fetchall()
    orphans = [r[0] for r in rows if r[0] not in active]
    for message_id in orphans:
        end_message(conn, message_id, "interrupted", INTERRUPTED_MESSAGE)
    return len(orphans)


# --- evidence ----------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkEvidence:
    chunk_id: int
    source_id: int
    source_filename: str
    page_no: int
    ord: int
    text: str
    prev_id: int | None
    next_id: int | None


def get_chunk_evidence(
    conn: sqlite3.Connection, notebook_id: int, chunk_id: int
) -> ChunkEvidence | None:
    """A live chunk of a ready source in this notebook, with neighbours in the
    same source only."""
    row = conn.execute(
        "SELECT c.id, c.source_id, c.page_no, c.ord, c.text, s.original_filename "
        "FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE c.id = ? AND c.notebook_id = ? AND s.notebook_id = ? "
        "AND s.status = 'ready'",
        (chunk_id, notebook_id, notebook_id),
    ).fetchone()
    if row is None:
        return None

    def neighbour(ord_: int) -> int | None:
        found = conn.execute(
            "SELECT id FROM chunks WHERE source_id = ? AND ord = ?",
            (row["source_id"], ord_),
        ).fetchone()
        return int(found[0]) if found else None

    return ChunkEvidence(
        chunk_id=row["id"],
        source_id=row["source_id"],
        source_filename=row["original_filename"],
        page_no=row["page_no"],
        ord=row["ord"],
        text=row["text"],
        prev_id=neighbour(row["ord"] - 1),
        next_id=neighbour(row["ord"] + 1),
    )
