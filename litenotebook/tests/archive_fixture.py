"""A realistic synthetic notebook for archive tests and smoke runs.

Two PDFs (chunks + embeddings from FakeEmbedder), a conversation whose answer
cites [S1] and a later answer that cites [M1], a correction, a note linked
to a source chunk, to an annotation and to a memory node, an annotation with
rectangles, and a saved reading position. No model is involved.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from conftest import FakeEmbedder, make_pdf
from litenotebook.chat import (
    CitedChunk,
    RememberedItem,
    complete_message,
    start_exchange,
)
from litenotebook.config import Settings
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.memory.graph import set_outcome, sync_turn
from litenotebook.memory.search import search_memory
from litenotebook.notebooks import create_notebook
from litenotebook.notes import (
    create_note,
    link_annotation,
    link_chunk,
    link_memory_node,
    link_message,
)
from litenotebook.reader.annotations import (
    create_annotation,
    open_source,
    save_reading_state,
)

PDF_A = [
    "Heliostats are mirrors that track the sun.",
    "Distinctive fact: the Borealis heliostat moves with an accuracy of 0.1 "
    "milliradians, which is remarkably precise.",
    "The receiver tower collects the reflected light.",
]
PDF_B = [
    "Molten salt stores heat for the night.",
    "Storage tanks keep the salt at 565 degrees Celsius.",
]


@dataclass
class Rich:
    notebook_id: int
    name: str
    sources: list[int]
    conversation_id: int
    answer_id: int
    second_answer_id: int
    citation_chunk: int
    note_id: int
    annotation_id: int


def build_rich_notebook(
    conn: sqlite3.Connection, settings: Settings, tmp: Path, name: str = "Solar"
) -> Rich:
    nb = create_notebook(conn, name, "Concentrated solar power <b>notes</b>")
    store = FaissStore(conn, settings)
    a = ingest_pdf(
        conn,
        settings,
        nb,
        make_pdf(tmp / f"{name}-a.pdf", PDF_A),
        FakeEmbedder(),
        store,
    )
    b = ingest_pdf(
        conn,
        settings,
        nb,
        make_pdf(tmp / f"{name}-b.pdf", PDF_B),
        FakeEmbedder(),
        store,
    )
    sources = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM sources WHERE notebook_id = ? ORDER BY id", (nb.id,)
        )
    ]
    chunk = conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks WHERE source_id = ? "
        "AND page_no = 2",
        (sources[0],),
    ).fetchone()
    del a, b

    ex = start_exchange(conn, nb.id, None, "How accurate is the Borealis heliostat?")
    complete_message(
        conn,
        ex.assistant_message_id,
        "It is accurate to 0.1 milliradians [S1].",
        model="synthetic",
        citations=[
            CitedChunk(
                1, chunk[0], chunk[1], f"{name}-a.pdf", chunk[2], chunk[3], chunk[4]
            )
        ],
    )
    sync_turn(conn, ex.assistant_message_id)  # as the chat service does
    set_outcome(
        conn,
        nb.id,
        ex.assistant_message_id,
        "corrected",
        "It is 0.1 mrad in calm wind only.",
    )
    found = search_memory(conn, nb.id, "Borealis heliostat accuracy")
    item = found.items[0]
    ex2 = start_exchange(conn, nb.id, ex.conversation.id, "And in strong wind?")
    complete_message(
        conn,
        ex2.assistant_message_id,
        "Earlier you noted calm wind only [M1].",
        model="synthetic",
        memory_refs=[
            RememberedItem(
                label=1,
                node_key=item.node_key,
                kind=item.kind,
                provenance=item.provenance,
                conversation_id=item.conversation_id,
                conversation_title=item.conversation_title,
                question=item.question,
                excerpt=item.excerpt,
                outcome=item.outcome,
                correction=item.correction,
                sources=item.sources,
                remembered_at=item.created_at,
            )
        ],
    )
    sync_turn(conn, ex2.assistant_message_id)

    reader = open_source(conn, settings, nb.id, sources[0])
    ann = create_annotation(
        conn,
        settings,
        nb.id,
        sources[0],
        page_no=2,
        selected_text="Borealis heliostat",
        comment="Key <i>fact</i>",
        color="green",
        rects=[[0.1, 0.05, 0.4, 0.07], [0.1, 0.08, 0.3, 0.1]],
    )
    save_reading_state(conn, reader, 2, 1.25)

    note = create_note(conn, nb.id, "Reading list", "Borealis heliostat accuracy notes")
    link_chunk(conn, nb.id, note.id, chunk[0])
    link_annotation(conn, nb.id, note.id, ann.id)
    link_message(conn, nb.id, note.id, ex.assistant_message_id)
    node = conn.execute(
        "SELECT id FROM memory_nodes WHERE notebook_id = ? AND key = ?",
        (nb.id, f"correction:{ex.assistant_message_id}"),
    ).fetchone()[0]
    link_memory_node(conn, nb.id, note.id, node)
    return Rich(
        nb.id,
        name,
        sources,
        ex.conversation.id,
        ex.assistant_message_id,
        ex2.assistant_message_id,
        chunk[0],
        note.id,
        ann.id,
    )


def _no_ids(details: str) -> str:
    data = json.loads(details)
    for c in data.get("citations", []):
        c.pop("chunk_id", None)
        c.pop("source_id", None)
    return json.dumps(data, sort_keys=True)


def logical_snapshot(conn: sqlite3.Connection, notebook_id: int) -> dict:
    """User-visible content without ids or timestamps of derived rows — equal
    before export and after import when nothing was lost."""
    q = lambda sql: [tuple(r) for r in conn.execute(sql, (notebook_id,))]  # noqa: E731
    return {
        "notebook": q(
            "SELECT name, description, embed_model, embed_dim, created_at "
            "FROM notebooks WHERE id = ?"
        ),
        "sources": q(
            "SELECT original_filename, sha256, size_bytes, page_count, status "
            "FROM sources WHERE notebook_id = ? ORDER BY sha256"
        ),
        "chunks": q(
            "SELECT s.sha256, c.ord, c.page_no, c.text, c.embedding FROM chunks c "
            "JOIN sources s ON s.id = c.source_id WHERE c.notebook_id = ? "
            "ORDER BY s.sha256, c.ord"
        ),
        "messages": q(
            "SELECT m.role, m.content, m.status, m.created_at FROM messages m "
            "JOIN conversations c ON c.id = m.conversation_id "
            "WHERE c.notebook_id = ? ORDER BY m.created_at, m.role"
        ),
        "citations": q(
            "SELECT mc.label, mc.source_filename, mc.page_no, mc.chunk_text, "
            "EXISTS (SELECT 1 FROM chunks ch WHERE ch.id = mc.chunk_id "
            "AND ch.source_id = mc.source_id) FROM message_citations mc "
            "JOIN messages m ON m.id = mc.message_id JOIN conversations c "
            "ON c.id = m.conversation_id WHERE c.notebook_id = ? "
            "ORDER BY mc.label"
        ),
        "memory_refs": q(
            "SELECT r.label, r.kind, r.question, r.excerpt, r.correction, "
            "EXISTS (SELECT 1 FROM memory_nodes n WHERE n.key = r.node_key "
            "AND n.notebook_id = c.notebook_id) FROM message_memory_refs r "
            "JOIN messages m ON m.id = r.message_id JOIN conversations c "
            "ON c.id = m.conversation_id WHERE c.notebook_id = ?"
        ),
        "outcomes": q(
            "SELECT outcome, correction FROM memory_outcomes WHERE notebook_id = ?"
        ),
        "notes": q("SELECT title, content, revision FROM notes WHERE notebook_id = ?"),
        "links": [
            (*r[:4], _no_ids(r[4]))
            for r in q(
                "SELECT l.link_type, l.label, l.excerpt, l.page_no, l.details, "
                "l.target_key FROM note_links l JOIN notes n ON n.id = l.note_id "
                "WHERE n.notebook_id = ? ORDER BY l.link_type"
            )
        ],
        "link_targets_live": q(
            "SELECT l.link_type, CASE l.link_type "
            "WHEN 'source_chunk' THEN EXISTS (SELECT 1 FROM chunks c "
            "  WHERE c.id = l.chunk_id) "
            "WHEN 'annotation' THEN EXISTS (SELECT 1 FROM annotations a "
            "  WHERE a.id = l.annotation_id) "
            "WHEN 'message' THEN EXISTS (SELECT 1 FROM messages m "
            "  WHERE m.id = l.message_id) "
            "ELSE EXISTS (SELECT 1 FROM memory_nodes m WHERE m.key = l.memory_key "
            "  AND m.notebook_id = n.notebook_id) END "
            "FROM note_links l JOIN notes n ON n.id = l.note_id "
            "WHERE n.notebook_id = ? "
            "ORDER BY l.link_type"
        ),
        "annotations": q(
            "SELECT a.page_no, a.selected_text, a.comment, a.color, "
            "(SELECT group_concat(x0 || ',' || y0 || ',' || x1 || ',' || y1, ';') "
            "FROM annotation_rects r WHERE r.annotation_id = a.id) "
            "FROM annotations a WHERE a.notebook_id = ?"
        ),
        "reading": q(
            "SELECT r.current_page, r.zoom FROM reading_state r WHERE r.notebook_id = ?"
        ),
        "memory": q(
            "SELECT node_type, count(*) FROM memory_nodes WHERE notebook_id = ? "
            "GROUP BY node_type ORDER BY node_type"
        ),
    }


def read_entries(path: Path) -> dict[str, bytes]:
    import zipfile

    with zipfile.ZipFile(path) as zf:
        return {i.filename: zf.read(i) for i in zf.infolist()}


def write_archive(
    path: Path,
    entries: dict[str, bytes] | list[tuple[str, bytes]],
    *,
    fix_checksums: bool = True,
    infos: dict[str, dict] | None = None,
) -> Path:
    """Write entries as a ZIP; by default recompute checksums.json and the
    manifest's checksums hash so only the intended defect remains."""
    import hashlib
    import warnings
    import zipfile

    items = list(entries.items()) if isinstance(entries, dict) else list(entries)
    if fix_checksums:
        names = [n for n, _ in items if n not in ("manifest.json", "checksums.json")]
        data = dict(items)
        sums = {n: hashlib.sha256(data[n]).hexdigest() for n in names}
        checksums = json.dumps({"algorithm": "sha256", "files": sums}).encode()
        manifest = json.loads(data["manifest.json"])
        manifest["checksums_sha256"] = hashlib.sha256(checksums).hexdigest()
        items = [
            (n, b) for n, b in items if n not in ("manifest.json", "checksums.json")
        ]
        items += [
            ("checksums.json", checksums),
            ("manifest.json", json.dumps(manifest).encode()),
        ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate names are written on purpose
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, blob in items:
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                for key, value in (infos or {}).get(name, {}).items():
                    setattr(info, key, value)
                zf.writestr(info, blob)
    return path


def edit_rows(entries: dict[str, bytes], name: str, fn) -> None:
    """Apply ``fn(list_of_rows)`` to one JSONL table in place."""
    rows = [json.loads(line) for line in entries[name].decode().splitlines() if line]
    fn(rows)
    entries[name] = "".join(json.dumps(r) + "\n" for r in rows).encode()
