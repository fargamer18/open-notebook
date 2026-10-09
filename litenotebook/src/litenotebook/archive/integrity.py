"""Integrity reports after export, import and restore (no models involved)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from litenotebook.config import Settings

FTS_TABLES = (
    "chunks_fts",
    "notes_fts",
    "messages_fts",
    "conversations_fts",
    "memory_nodes_fts",
    "annotations_fts",
)


@dataclass
class NotebookReport:
    notebook_id: int
    name: str
    counts: dict[str, int]
    faiss: str  # e.g. "125/125"
    faiss_ok: bool
    memory_ok: bool
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "counts": self.counts,
            "faiss": self.faiss,
            "faiss_ok": self.faiss_ok,
            "memory_ok": self.memory_ok,
            "problems": self.problems,
        }


@dataclass
class DatabaseReport:
    integrity: str  # "ok" or SQLite's first message
    foreign_keys: int  # violating rows
    fts_problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.integrity == "ok" and self.foreign_keys == 0 and not self.fts_problems
        )


def database_report(conn: sqlite3.Connection) -> DatabaseReport:
    integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    fk = len(conn.execute("PRAGMA foreign_key_check").fetchall())
    problems = []
    for table in FTS_TABLES:
        try:
            conn.execute(f"INSERT INTO {table}({table}) VALUES('integrity-check')")
        except sqlite3.DatabaseError as e:
            problems.append(f"{table}: {e}")
    if conn.in_transaction:
        conn.commit()
    return DatabaseReport(integrity, fk, problems)


def notebook_counts(conn: sqlite3.Connection, notebook_id: int) -> dict[str, int]:
    def one(sql: str) -> int:
        return int(conn.execute(sql, (notebook_id,)).fetchone()[0])

    return {
        "sources": one("SELECT count(*) FROM sources WHERE notebook_id = ?"),
        "ready_sources": one(
            "SELECT count(*) FROM sources WHERE notebook_id = ? AND status = 'ready'"
        ),
        "pages": one(
            "SELECT count(*) FROM pages p JOIN sources s ON s.id = p.source_id "
            "WHERE s.notebook_id = ?"
        ),
        "chunks": one("SELECT count(*) FROM chunks WHERE notebook_id = ?"),
        "embedded_chunks": one(
            "SELECT count(*) FROM chunks WHERE notebook_id = ? "
            "AND embedding IS NOT NULL"
        ),
        "notes": one("SELECT count(*) FROM notes WHERE notebook_id = ?"),
        "conversations": one(
            "SELECT count(*) FROM conversations WHERE notebook_id = ?"
        ),
        "messages": one(
            "SELECT count(*) FROM messages m JOIN conversations c "
            "ON c.id = m.conversation_id WHERE c.notebook_id = ?"
        ),
        "citations": one(
            "SELECT count(*) FROM message_citations mc JOIN messages m "
            "ON m.id = mc.message_id JOIN conversations c ON c.id = m.conversation_id "
            "WHERE c.notebook_id = ?"
        ),
        "memory_refs": one(
            "SELECT count(*) FROM message_memory_refs r JOIN messages m "
            "ON m.id = r.message_id JOIN conversations c ON c.id = m.conversation_id "
            "WHERE c.notebook_id = ?"
        ),
        "outcomes": one("SELECT count(*) FROM memory_outcomes WHERE notebook_id = ?"),
        "corrections": one(
            "SELECT count(*) FROM memory_outcomes WHERE notebook_id = ? "
            "AND outcome = 'corrected'"
        ),
        "annotations": one("SELECT count(*) FROM annotations WHERE notebook_id = ?"),
        "memory_nodes": one("SELECT count(*) FROM memory_nodes WHERE notebook_id = ?"),
        "memory_edges": one("SELECT count(*) FROM memory_edges WHERE notebook_id = ?"),
        "quizzes": one("SELECT count(*) FROM quizzes WHERE notebook_id = ?"),
        "quiz_questions": one(
            "SELECT count(*) FROM quiz_questions q JOIN quizzes z ON z.id = q.quiz_id "
            "WHERE z.notebook_id = ?"
        ),
        "quiz_evidence": one(
            "SELECT count(*) FROM quiz_evidence e JOIN quiz_questions q "
            "ON q.id = e.question_id JOIN quizzes z ON z.id = q.quiz_id "
            "WHERE z.notebook_id = ?"
        ),
        "quiz_attempts": one(
            "SELECT count(*) FROM quiz_attempts a JOIN quizzes z ON z.id = a.quiz_id "
            "WHERE z.notebook_id = ?"
        ),
        "quiz_answers": one(
            "SELECT count(*) FROM quiz_attempt_items i JOIN quiz_attempts a "
            "ON a.id = i.attempt_id JOIN quizzes z ON z.id = a.quiz_id "
            "WHERE z.notebook_id = ? AND i.answered_at IS NOT NULL"
        ),
        "study_concepts": one(
            "SELECT count(*) FROM study_concept_stats WHERE notebook_id = ?"
        ),
    }


def notebook_report(
    conn: sqlite3.Connection, settings: Settings, notebook_id: int
) -> NotebookReport:
    from litenotebook.index.faiss_store import FaissStore
    from litenotebook.memory.graph import check as memory_check

    name = conn.execute(
        "SELECT name FROM notebooks WHERE id = ?", (notebook_id,)
    ).fetchone()[0]
    counts = notebook_counts(conn, notebook_id)
    problems: list[str] = []
    status = FaissStore(conn, settings).check(notebook_id)
    index = status.index_count if status.index_count is not None else 0
    faiss = f"{index}/{status.sqlite_count}"
    problems += [f"FAISS: {p}" for p in status.problems]
    memory = memory_check(conn, notebook_id)
    problems += [f"Memory: {p}" for p in memory.problems]
    from litenotebook.study.store import stats_consistent

    if not stats_consistent(conn, notebook_id):
        problems.append("Study: concept progress does not match the quiz answers")
    for sid, filename in conn.execute(
        "SELECT id, original_filename FROM sources WHERE notebook_id = ?",
        (notebook_id,),
    ).fetchall():
        path = settings.source_file(notebook_id, sid)
        if path.is_symlink() or not path.is_file():
            problems.append(f"Source file missing: {filename}")
    return NotebookReport(
        notebook_id=notebook_id,
        name=name,
        counts=counts,
        faiss=faiss,
        faiss_ok=status.ok,
        memory_ok=not memory.problems,
        problems=problems,
    )
