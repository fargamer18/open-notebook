"""Notes: user-authored text per notebook, with snapshot links to evidence.

* Notes are plain text (never rendered as HTML).
* Saves carry the revision the editor started from; a stale revision is a
  conflict instead of a silent overwrite (two tabs, a stale autosave).
* Links point at application-controlled targets — a source chunk, a citation
  snapshot, a chat message or a memory node — and store an immutable snapshot,
  so a note stays readable after the linked item is deleted.
* Every save re-derives the note's graph memory (deterministic, no LLM).

No heavy imports: notes never load MiniLM, FAISS or torch.
"""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from typing import Any

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
MAX_TITLE_CHARS = 200
MAX_CONTENT_BYTES = 1024 * 1024
EXCERPT_CHARS = 600
LINK_TYPES = ("source_chunk", "message", "memory_node", "annotation")


class NoteError(ValueError):
    pass


class NoteNotFoundError(NoteError):
    pass


class NoteConflictError(NoteError):
    pass


@dataclass(frozen=True)
class Note:
    id: int
    notebook_id: int
    title: str
    content: str
    revision: int
    created_at: str
    updated_at: str
    link_count: int = 0


@dataclass
class NoteLink:
    id: int
    note_id: int
    link_type: str
    target_key: str
    label: str
    excerpt: str
    source_id: int | None
    chunk_id: int | None
    message_id: int | None
    conversation_id: int | None
    memory_key: str | None
    page_no: int | None
    chunk_ord: int | None
    details: dict[str, Any]
    created_at: str
    exists: bool = True  # filled in at read time
    source_live: bool = False  # the original PDF can still be opened
    annotation_id: int | None = None
    memory_node_id: int | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def clean_title(title: str) -> str:
    title = " ".join((title or "").split())
    title = "".join(ch for ch in title if unicodedata.category(ch) not in ("Cc", "Cf"))
    if not title:
        raise NoteError("The note needs a title")
    if len(title) > MAX_TITLE_CHARS:
        raise NoteError(f"Titles are limited to {MAX_TITLE_CHARS} characters")
    return title


def clean_content(content: str) -> str:
    content = (content or "").replace("\r\n", "\n").replace("\x00", "")
    if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise NoteError("Notes are limited to 1 MiB of text")
    return content


def _note(row: sqlite3.Row) -> Note:
    return Note(
        id=row["id"],
        notebook_id=row["notebook_id"],
        title=row["title"],
        content=row["content"],
        revision=row["revision"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        link_count=row["n_links"] if "n_links" in row.keys() else 0,
    )


_SELECT = (
    "SELECT n.*, (SELECT count(*) FROM note_links l WHERE l.note_id = n.id) AS n_links "
    "FROM notes n"
)


def list_notes(conn: sqlite3.Connection, notebook_id: int) -> list[Note]:
    rows = conn.execute(
        _SELECT + " WHERE n.notebook_id = ? ORDER BY n.updated_at DESC, n.id DESC",
        (notebook_id,),
    ).fetchall()
    return [_note(r) for r in rows]


def get_note(conn: sqlite3.Connection, notebook_id: int, note_id: int) -> Note:
    row = conn.execute(
        _SELECT + " WHERE n.id = ? AND n.notebook_id = ?", (note_id, notebook_id)
    ).fetchone()
    if row is None:
        raise NoteNotFoundError(f"Note {note_id} not found in this notebook")
    return _note(row)


def _sync(conn: sqlite3.Connection, note_id: int) -> None:
    from litenotebook.memory.graph import sync_note

    sync_note(conn, note_id)


def create_note(
    conn: sqlite3.Connection, notebook_id: int, title: str, content: str = ""
) -> Note:
    title, content = clean_title(title), clean_content(content)
    with conn:
        cur = conn.execute(
            "INSERT INTO notes (notebook_id, title, content) VALUES (?, ?, ?)",
            (notebook_id, title, content),
        )
    assert cur.lastrowid is not None
    _sync(conn, cur.lastrowid)
    return get_note(conn, notebook_id, cur.lastrowid)


def update_note(
    conn: sqlite3.Connection,
    notebook_id: int,
    note_id: int,
    *,
    title: str | None = None,
    content: str | None = None,
    base_revision: int | None = None,
) -> Note:
    note = get_note(conn, notebook_id, note_id)
    new_title = clean_title(title) if title is not None else note.title
    new_content = clean_content(content) if content is not None else note.content
    if base_revision is not None and base_revision != note.revision:
        raise NoteConflictError(
            "This note was changed elsewhere (another tab?). Reload it before "
            "saving so nothing is overwritten."
        )
    if new_title == note.title and new_content == note.content:
        return note
    with conn:
        cur = conn.execute(
            f"UPDATE notes SET title = ?, content = ?, revision = revision + 1, "
            f"updated_at = {NOW_SQL} WHERE id = ? AND notebook_id = ? AND revision = ?",
            (new_title, new_content, note_id, notebook_id, note.revision),
        )
        if cur.rowcount != 1:
            raise NoteConflictError("This note was changed elsewhere; reload it")
    _sync(conn, note_id)
    return get_note(conn, notebook_id, note_id)


def append_to_note(
    conn: sqlite3.Connection, notebook_id: int, note_id: int, text: str
) -> Note:
    note = get_note(conn, notebook_id, note_id)
    joined = (
        f"{note.content.rstrip()}\n\n{text.strip()}" if note.content.strip() else text
    )
    return update_note(conn, notebook_id, note_id, content=joined)


def delete_note(conn: sqlite3.Connection, notebook_id: int, note_id: int) -> Note:
    note = get_note(conn, notebook_id, note_id)
    with conn:
        conn.execute(
            "DELETE FROM notes WHERE id = ? AND notebook_id = ?", (note_id, notebook_id)
        )
        # its memory nodes cascade; drop concepts nothing else mentions
        from litenotebook.memory.graph import prune_orphan_concepts

        prune_orphan_concepts(conn, notebook_id)
    return note


# --- links ------------------------------------------------------------------------


def _insert_link(
    conn: sqlite3.Connection, note_id: int, link_type: str, target_key: str, **v: Any
) -> int:
    assert link_type in LINK_TYPES
    with conn:
        conn.execute(
            "INSERT INTO note_links (note_id, link_type, target_key, source_id, "
            "chunk_id, "
            "message_id, conversation_id, memory_key, annotation_id, label, excerpt, "
            "page_no, chunk_ord, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(note_id, target_key) DO NOTHING",
            (
                note_id,
                link_type,
                target_key,
                v.get("source_id"),
                v.get("chunk_id"),
                v.get("message_id"),
                v.get("conversation_id"),
                v.get("memory_key"),
                v.get("annotation_id"),
                v["label"],
                v.get("excerpt", ""),
                v.get("page_no"),
                v.get("chunk_ord"),
                json.dumps(v.get("details", {})),
            ),
        )
        conn.execute(
            f"UPDATE notes SET updated_at = {NOW_SQL} WHERE id = ?", (note_id,)
        )
    row = conn.execute(
        "SELECT id FROM note_links WHERE note_id = ? AND target_key = ?",
        (note_id, target_key),
    ).fetchone()
    _sync(conn, note_id)
    return int(row[0])


def link_chunk(
    conn: sqlite3.Connection, notebook_id: int, note_id: int, chunk_id: int
) -> int:
    """Link a live chunk of a ready source in this notebook (snapshot kept)."""
    get_note(conn, notebook_id, note_id)
    row = conn.execute(
        "SELECT c.id, c.source_id, c.page_no, c.ord, c.text, s.original_filename "
        "FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE c.id = ? AND c.notebook_id = ? AND s.notebook_id = ? "
        "AND s.status = 'ready'",
        (chunk_id, notebook_id, notebook_id),
    ).fetchone()
    if row is None:
        raise NoteNotFoundError("That passage is not in this notebook")
    return _insert_link(
        conn,
        note_id,
        "source_chunk",
        f"chunk:{row['id']}",
        source_id=row["source_id"],
        chunk_id=row["id"],
        page_no=row["page_no"],
        chunk_ord=row["ord"],
        label=row["original_filename"],
        excerpt=row["text"],
    )


def link_citation(
    conn: sqlite3.Connection, notebook_id: int, note_id: int, citation_id: int
) -> int:
    """Link the evidence behind an [S#] (works even if its source was deleted)."""
    from litenotebook.chat import get_citation

    get_note(conn, notebook_id, note_id)
    c = get_citation(conn, notebook_id, citation_id)
    return _insert_link(
        conn,
        note_id,
        "source_chunk",
        f"chunk:{c.chunk_id}",
        source_id=c.source_id,
        chunk_id=c.chunk_id,
        page_no=c.page_no,
        chunk_ord=c.chunk_ord,
        label=c.source_filename,
        excerpt=c.chunk_text,
    )


def link_message(
    conn: sqlite3.Connection, notebook_id: int, note_id: int, message_id: int
) -> int:
    from litenotebook.chat import get_conversation, get_message

    get_note(conn, notebook_id, note_id)
    m = get_message(conn, notebook_id, message_id)
    conversation = get_conversation(conn, notebook_id, m.conversation_id)
    citations = [
        {
            "label": c.label,
            "filename": c.source_filename,
            "page": c.page_no,
            "chunk": c.chunk_ord,
            "chunk_id": c.chunk_id,
            "source_id": c.source_id,
        }
        for c in m.citations
    ]
    memory = [
        {"label": r.label, "trust": r.trust, "title": r.conversation_title}
        for r in m.memory_refs
    ]
    return _insert_link(
        conn,
        note_id,
        "message",
        f"message:{m.id}",
        message_id=m.id,
        conversation_id=m.conversation_id,
        label=conversation.title,
        excerpt=_clip(m.content, EXCERPT_CHARS),
        details={"role": m.role, "citations": citations, "memory": memory},
    )


def link_memory_node(
    conn: sqlite3.Connection, notebook_id: int, note_id: int, node_id: int
) -> int:
    get_note(conn, notebook_id, note_id)
    row = conn.execute(
        "SELECT key, node_type, label, content, note_id FROM memory_nodes "
        "WHERE id = ? AND notebook_id = ?",
        (node_id, notebook_id),
    ).fetchone()
    if row is None:
        raise NoteNotFoundError("That memory item is not in this notebook")
    if row["note_id"] == note_id:
        raise NoteError("A note cannot link to itself")
    return _insert_link(
        conn,
        note_id,
        "memory_node",
        f"memory:{row['key']}",
        memory_key=row["key"],
        label=row["label"],
        excerpt=_clip(row["content"], EXCERPT_CHARS),
        details={"node_type": row["node_type"]},
    )


def link_annotation(
    conn: sqlite3.Connection, notebook_id: int, note_id: int, annotation_id: int
) -> int:
    """Link a reader highlight/annotation (snapshot of text, comment, page)."""
    from litenotebook.reader.annotations import get_annotation

    get_note(conn, notebook_id, note_id)
    a = get_annotation(conn, notebook_id, annotation_id)
    return _insert_link(
        conn,
        note_id,
        "annotation",
        f"annotation:{a.id}",
        source_id=a.source_id,
        annotation_id=a.id,
        page_no=a.page_no,
        label=a.source_filename,
        excerpt=a.selected_text,
        details={"comment": a.comment, "color": a.color, "type": a.annotation_type},
    )


def remove_link(
    conn: sqlite3.Connection, notebook_id: int, note_id: int, link_id: int
) -> None:
    get_note(conn, notebook_id, note_id)
    with conn:
        conn.execute(
            "DELETE FROM note_links WHERE id = ? AND note_id = ?", (link_id, note_id)
        )
    _sync(conn, note_id)


def list_links(
    conn: sqlite3.Connection, notebook_id: int, note_id: int
) -> list[NoteLink]:
    """Links with their liveness decided now (never relinked elsewhere)."""
    get_note(conn, notebook_id, note_id)
    links: list[NoteLink] = []
    for r in conn.execute(
        "SELECT * FROM note_links WHERE note_id = ? ORDER BY id", (note_id,)
    ):
        try:
            details = json.loads(r["details"] or "{}")
        except ValueError:
            details = {}
        link = NoteLink(
            id=r["id"],
            note_id=r["note_id"],
            link_type=r["link_type"],
            target_key=r["target_key"],
            label=r["label"],
            excerpt=r["excerpt"],
            source_id=r["source_id"],
            chunk_id=r["chunk_id"],
            message_id=r["message_id"],
            conversation_id=r["conversation_id"],
            memory_key=r["memory_key"],
            page_no=r["page_no"],
            chunk_ord=r["chunk_ord"],
            details=details if isinstance(details, dict) else {},
            created_at=r["created_at"],
            annotation_id=r["annotation_id"],
        )
        if link.source_id is not None:
            link.source_live = (
                conn.execute(
                    "SELECT 1 FROM sources WHERE id = ? AND notebook_id = ?",
                    (link.source_id, notebook_id),
                ).fetchone()
                is not None
            )
        if link.link_type == "annotation":
            link.exists = (
                conn.execute(
                    "SELECT 1 FROM annotations WHERE id = ? AND notebook_id = ?",
                    (link.annotation_id, notebook_id),
                ).fetchone()
                is not None
            )
        elif link.link_type == "source_chunk":
            link.exists = (
                conn.execute(
                    "SELECT 1 FROM chunks WHERE id = ? AND source_id = ? "
                    "AND notebook_id = ?",
                    (link.chunk_id, link.source_id, notebook_id),
                ).fetchone()
                is not None
            )
        elif link.link_type == "message":
            link.exists = (
                conn.execute(
                    "SELECT 1 FROM messages m "
                    "JOIN conversations c ON c.id = m.conversation_id "
                    "WHERE m.id = ? AND c.notebook_id = ?",
                    (link.message_id, notebook_id),
                ).fetchone()
                is not None
            )
            link.citations = list(link.details.get("citations", []))
        else:
            node = conn.execute(
                "SELECT id FROM memory_nodes WHERE notebook_id = ? AND key = ?",
                (notebook_id, link.memory_key),
            ).fetchone()
            link.exists = node is not None
            link.memory_node_id = int(node[0]) if node else None
        links.append(link)
    return links


def note_concepts(
    conn: sqlite3.Connection, notebook_id: int, note_id: int
) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT c.label FROM memory_nodes n "
            "JOIN memory_edges e ON e.from_node_id = n.id "
            "JOIN memory_nodes c ON c.id = e.to_node_id "
            "WHERE n.notebook_id = ? AND n.key = ? AND e.relation = 'note_mentions' "
            "ORDER BY c.label",
            (notebook_id, f"note:{note_id}"),
        )
    ]


def note_title_for_answer(question: str) -> str:
    from litenotebook.chat import title_from

    return clean_title(f"Saved answer: {title_from(question)}")
