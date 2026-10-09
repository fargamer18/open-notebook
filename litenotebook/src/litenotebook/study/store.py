"""Quizzes, attempts and study progress in SQLite (see migration 0009).

Rendering Study pages, taking a quiz and reviewing it use only this module:
no retrieval, no MiniLM and no LLM. Every lookup is scoped by notebook id,
so ids from another notebook are "not found".

Attempt lifecycle::

    start (full | mistakes) → items created in display order, in_progress
    answer an item once     → selection + correctness fixed (trigger-enforced)
    last item answered      → completed, score = correct items
    finish early            → completed; unanswered items count as not correct
    start a new full attempt while one is open → the open one is abandoned

Concept stats are derived from answered items and updated as each answer is
recorded; ``rebuild_concept_stats`` recomputes them from scratch (used after
a quiz is deleted and to check an imported notebook).
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from litenotebook.study.model import (
    DIFFICULTY_LABELS,
    INTERRUPTED,
    KIND_PREFIX,
    QTYPE_LABELS,
    QuizQuestion,
)

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


class StudyError(ValueError):
    pass


class QuizNotFoundError(StudyError):
    pass


class AttemptNotFoundError(StudyError):
    pass


class AlreadyAnsweredError(StudyError):
    pass


# --- read models -------------------------------------------------------------------


@dataclass
class Evidence:
    ordinal: int
    kind: str
    label: str  # file name / note title / conversation title
    excerpt: str
    source_id: int | None
    chunk_id: int | None
    page_no: int | None
    chunk_ord: int | None
    note_id: int | None
    message_id: int | None
    conversation_id: int | None
    exists: bool = False  # the source / note / conversation still exists
    chunk_exists: bool = False
    number: int = 0  # 1-based within its kind for this question: S1, S2, N1 …

    @property
    def tag(self) -> str:
        return f"{KIND_PREFIX[self.kind]}{self.number or self.ordinal}"


@dataclass
class Option:
    position: int
    text: str
    is_correct: bool

    @property
    def letter(self) -> str:
        return "ABCD"[self.position - 1]


@dataclass
class Question:
    id: int
    position: int
    qtype: str
    question: str
    explanation: str
    hint: str
    difficulty: str
    concepts: list[dict[str, str]]
    options: list[Option] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def type_label(self) -> str:
        return QTYPE_LABELS[self.qtype]

    @property
    def correct_positions(self) -> list[int]:
        return [o.position for o in self.options if o.is_correct]


@dataclass
class Quiz:
    id: int
    notebook_id: int
    title: str
    status: str
    stage: str
    difficulty: str
    question_types: list[str]
    requested_count: int
    question_count: int
    focus: str
    material: dict[str, Any]
    message: str | None
    error: str | None
    model: str | None
    meta: dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: str | None
    attempts: int = 0
    best: int | None = None
    latest: tuple[int, int] | None = None  # (score, total) of the latest completed
    open_attempt: int | None = None

    @property
    def difficulty_label(self) -> str:
        return DIFFICULTY_LABELS.get(self.difficulty, self.difficulty)

    @property
    def type_labels(self) -> list[str]:
        return [QTYPE_LABELS[t] for t in self.question_types if t in QTYPE_LABELS]

    @property
    def source_names(self) -> list[str]:
        return [s.get("filename", "") for s in self.material.get("sources", [])]


@dataclass
class Item:
    id: int
    position: int
    question_id: int
    selected: list[int] | None
    is_correct: bool | None
    hint_used: bool
    answered_at: str | None

    @property
    def answered(self) -> bool:
        return self.answered_at is not None


@dataclass
class Attempt:
    id: int
    quiz_id: int
    mode: str
    parent_attempt_id: int | None
    status: str
    total: int
    score: int | None
    started_at: str
    completed_at: str | None
    items: list[Item] = field(default_factory=list)

    @property
    def answered(self) -> int:
        return sum(1 for i in self.items if i.answered)

    @property
    def correct(self) -> int:
        return sum(1 for i in self.items if i.is_correct)

    @property
    def incorrect(self) -> int:
        return sum(1 for i in self.items if i.answered and not i.is_correct)

    @property
    def skipped(self) -> int:
        return sum(1 for i in self.items if not i.answered)

    @property
    def missed(self) -> list[Item]:
        return [i for i in self.items if not i.is_correct]


def _quiz(row: sqlite3.Row) -> Quiz:
    return Quiz(
        id=row["id"],
        notebook_id=row["notebook_id"],
        title=row["title"],
        status=row["status"],
        stage=row["stage"],
        difficulty=row["difficulty"],
        question_types=json.loads(row["question_types"] or "[]"),
        requested_count=row["requested_count"],
        question_count=row["question_count"],
        focus=row["focus"],
        material=json.loads(row["material"] or "{}"),
        message=row["message"],
        error=row["error"],
        model=row["model"],
        meta=json.loads(row["meta"] or "{}"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


_QUIZ_STATS = """
    SELECT q.*,
      (SELECT count(*) FROM quiz_attempts a WHERE a.quiz_id = q.id
         AND a.status = 'completed') AS n_attempts,
      (SELECT max(a.score) FROM quiz_attempts a WHERE a.quiz_id = q.id
         AND a.status = 'completed' AND a.mode = 'full') AS best,
      (SELECT a.score || '/' || a.total FROM quiz_attempts a WHERE a.quiz_id = q.id
         AND a.status = 'completed' AND a.mode = 'full'
         ORDER BY a.completed_at DESC, a.id DESC LIMIT 1) AS latest,
      (SELECT a.id FROM quiz_attempts a WHERE a.quiz_id = q.id
         AND a.status = 'in_progress' ORDER BY a.id DESC LIMIT 1) AS open_attempt
    FROM quizzes q
"""


def _with_stats(row: sqlite3.Row) -> Quiz:
    quiz = _quiz(row)
    quiz.attempts = row["n_attempts"]
    quiz.best = row["best"]
    if row["latest"]:
        score, total = row["latest"].split("/")
        quiz.latest = (int(score), int(total))
    quiz.open_attempt = row["open_attempt"]
    return quiz


def list_quizzes(
    conn: sqlite3.Connection, notebook_id: int, limit: int = 200
) -> list[Quiz]:
    rows = conn.execute(
        _QUIZ_STATS + " WHERE q.notebook_id = ? ORDER BY q.created_at DESC, q.id DESC "
        "LIMIT ?",
        (notebook_id, limit),
    ).fetchall()
    return [_with_stats(r) for r in rows]


def get_quiz(conn: sqlite3.Connection, notebook_id: int, quiz_id: int) -> Quiz:
    row = conn.execute(
        _QUIZ_STATS + " WHERE q.id = ? AND q.notebook_id = ?", (quiz_id, notebook_id)
    ).fetchone()
    if row is None:
        raise QuizNotFoundError(f"Quiz {quiz_id} not found in this notebook")
    return _with_stats(row)


def latest_settings(
    conn: sqlite3.Connection, notebook_id: int
) -> dict[str, Any] | None:
    """The choices of this notebook's most recent quiz (form defaults)."""
    row = conn.execute(
        "SELECT difficulty, question_types, requested_count, material FROM quizzes "
        "WHERE notebook_id = ? ORDER BY id DESC LIMIT 1",
        (notebook_id,),
    ).fetchone()
    if row is None:
        return None
    material = json.loads(row["material"] or "{}")
    return {
        "difficulty": row["difficulty"],
        "types": json.loads(row["question_types"]),
        "count": row["requested_count"],
        "notes": bool(material.get("notes")),
        "chat": bool(material.get("chat")),
    }


def questions(
    conn: sqlite3.Connection, quiz_id: int, ids: list[int] | None = None
) -> list[Question]:
    """Questions in quiz order with options and evidence (live-state flags
    tell the UI whether "Open in PDF" can work)."""
    rows = conn.execute(
        "SELECT * FROM quiz_questions WHERE quiz_id = ? ORDER BY position", (quiz_id,)
    ).fetchall()
    out = {
        r["id"]: Question(
            id=r["id"],
            position=r["position"],
            qtype=r["qtype"],
            question=r["question"],
            explanation=r["explanation"],
            hint=r["hint"],
            difficulty=r["difficulty"],
            concepts=json.loads(r["concepts"] or "[]"),
        )
        for r in rows
        if ids is None or r["id"] in ids
    }
    if not out:
        return []
    marks = ",".join("?" * len(out))
    for r in conn.execute(
        f"SELECT question_id, position, text, is_correct FROM quiz_question_options "
        f"WHERE question_id IN ({marks}) ORDER BY question_id, position",
        list(out),
    ):
        out[r["question_id"]].options.append(
            Option(r["position"], r["text"], bool(r["is_correct"]))
        )
    for r in conn.execute(
        f"""SELECT e.*,
              CASE e.kind
                WHEN 'source' THEN EXISTS (SELECT 1 FROM sources s
                     WHERE s.id = e.source_id AND s.status = 'ready'
                     AND s.notebook_id = q.notebook_id)
                WHEN 'note' THEN EXISTS (SELECT 1 FROM notes n
                     WHERE n.id = e.note_id AND n.notebook_id = q.notebook_id)
                ELSE EXISTS (SELECT 1 FROM conversations c
                     WHERE c.id = e.conversation_id AND c.notebook_id = q.notebook_id)
              END AS live,
              EXISTS (SELECT 1 FROM chunks c WHERE c.id = e.chunk_id
                     AND c.source_id = e.source_id) AS chunk_live
            FROM quiz_evidence e JOIN quiz_questions qq ON qq.id = e.question_id
            JOIN quizzes q ON q.id = qq.quiz_id
            WHERE e.question_id IN ({marks}) ORDER BY e.question_id, e.ordinal""",
        list(out),
    ):
        out[r["question_id"]].evidence.append(
            Evidence(
                ordinal=r["ordinal"],
                kind=r["kind"],
                label=r["label"],
                excerpt=r["excerpt"],
                source_id=r["source_id"],
                chunk_id=r["chunk_id"],
                page_no=r["page_no"],
                chunk_ord=r["chunk_ord"],
                note_id=r["note_id"],
                message_id=r["message_id"],
                conversation_id=r["conversation_id"],
                exists=bool(r["live"]),
                chunk_exists=bool(r["chunk_live"]),
            )
        )
    for q in out.values():
        seen: dict[str, int] = {}
        for e in q.evidence:
            seen[e.kind] = e.number = seen.get(e.kind, 0) + 1
    return sorted(out.values(), key=lambda q: q.position)


# --- creating quizzes --------------------------------------------------------------


def create_quiz(
    conn: sqlite3.Connection,
    notebook_id: int,
    *,
    title: str,
    difficulty: str,
    types: list[str],
    count: int,
    focus: str,
    material: dict[str, Any],
) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO quizzes (notebook_id, title, difficulty, question_types, "
            "requested_count, focus, material, stage) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                notebook_id,
                title,
                difficulty,
                json.dumps(types),
                count,
                focus,
                json.dumps(material, ensure_ascii=False),
                "queued",
            ),
        )
    assert cur.lastrowid is not None
    return cur.lastrowid


def set_stage(conn: sqlite3.Connection, quiz_id: int, stage: str) -> None:
    with conn:
        conn.execute(
            f"UPDATE quizzes SET stage = ?, updated_at = {NOW_SQL} WHERE id = ? "
            "AND status = 'generating'",
            (stage, quiz_id),
        )


def finish_failed(
    conn: sqlite3.Connection,
    quiz_id: int,
    error: str,
    *,
    status: str = "failed",
    meta: dict[str, Any] | None = None,
) -> None:
    with conn:
        conn.execute("DELETE FROM quiz_questions WHERE quiz_id = ?", (quiz_id,))
        conn.execute(
            f"UPDATE quizzes SET status = ?, error = ?, stage = ?, question_count = 0, "
            f"meta = ?, updated_at = {NOW_SQL}, completed_at = {NOW_SQL} WHERE id = ?",
            (status, error, status, json.dumps(meta or {}), quiz_id),
        )


def _shuffled(question: QuizQuestion, seed: str) -> tuple[list[str], list[int]]:
    """Option order is shuffled (deterministically) so the model's habit of
    putting the answer first does not show. True/False keeps its order."""
    if question.qtype == "true_false":
        return question.options, question.correct
    order = list(range(len(question.options)))
    random.Random(seed).shuffle(order)
    options = [question.options[i] for i in order]
    correct = sorted(order.index(i) for i in question.correct)
    return options, correct


def save_questions(
    conn: sqlite3.Connection,
    quiz_id: int,
    found: list[QuizQuestion],
    *,
    message: str | None,
    model: str | None,
    meta: dict[str, Any],
) -> None:
    """Persist validated questions and mark the quiz ready, in one transaction."""
    with conn:
        status = conn.execute(
            "SELECT status FROM quizzes WHERE id = ?", (quiz_id,)
        ).fetchone()
        if status is None or status[0] != "generating":
            raise StudyError("This quiz is no longer being generated")
        for position, q in enumerate(found, 1):
            cur = conn.execute(
                "INSERT INTO quiz_questions (quiz_id, position, qtype, question, "
                "explanation, hint, difficulty, concepts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    quiz_id,
                    position,
                    q.qtype,
                    q.question,
                    q.explanation,
                    q.hint,
                    q.difficulty,
                    json.dumps(
                        [{"key": c.key, "label": c.label} for c in q.concepts],
                        ensure_ascii=False,
                    ),
                ),
            )
            qid = cur.lastrowid
            options, correct = _shuffled(q, f"{quiz_id}:{position}:{q.question}")
            conn.executemany(
                "INSERT INTO quiz_question_options (question_id, position, text, "
                "is_correct) VALUES (?, ?, ?, ?)",
                [
                    (qid, i + 1, text, int(i in correct))
                    for i, text in enumerate(options)
                ],
            )
            for ordinal, e in enumerate(q.evidence, 1):
                conn.execute(
                    "INSERT INTO quiz_evidence (question_id, ordinal, kind, source_id, "
                    "chunk_id, page_no, chunk_ord, note_id, message_id, "
                    "conversation_id, label, excerpt) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        qid,
                        ordinal,
                        e.kind,
                        e.source_id,
                        e.chunk_id,
                        e.page_no,
                        e.chunk_ord,
                        e.note_id,
                        e.message_id,
                        e.conversation_id,
                        e.title,
                        e.text,
                    ),
                )
        conn.execute(
            "UPDATE quizzes SET status = 'ready', stage = 'ready', "
            "question_count = ?, message = ?, model = ?, meta = ?, "
            f"updated_at = {NOW_SQL}, completed_at = {NOW_SQL} WHERE id = ?",
            (len(found), message, model, json.dumps(meta), quiz_id),
        )


def recover_generating(conn: sqlite3.Connection) -> int:
    """Startup: a quiz left 'generating' by a previous process failed."""
    ids = [
        r[0] for r in conn.execute("SELECT id FROM quizzes WHERE status = 'generating'")
    ]
    for quiz_id in ids:
        finish_failed(conn, quiz_id, INTERRUPTED)
    return len(ids)


def delete_quiz(conn: sqlite3.Connection, notebook_id: int, quiz_id: int) -> Quiz:
    """Delete a quiz with its questions, evidence snapshots and every attempt.
    Concept stats are recomputed from the remaining attempts."""
    quiz = get_quiz(conn, notebook_id, quiz_id)
    if quiz.status == "generating":
        raise StudyError("This quiz is still being generated; cancel it first")
    with conn:
        conn.execute("DELETE FROM quizzes WHERE id = ?", (quiz_id,))
        _rebuild_stats(conn, notebook_id)
    return quiz


# --- attempts ----------------------------------------------------------------------


def _attempt(conn: sqlite3.Connection, row: sqlite3.Row) -> Attempt:
    attempt = Attempt(
        id=row["id"],
        quiz_id=row["quiz_id"],
        mode=row["mode"],
        parent_attempt_id=row["parent_attempt_id"],
        status=row["status"],
        total=row["total"],
        score=row["score"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )
    attempt.items = [
        Item(
            id=r["id"],
            position=r["position"],
            question_id=r["question_id"],
            selected=json.loads(r["selected"]) if r["selected"] is not None else None,
            is_correct=None if r["is_correct"] is None else bool(r["is_correct"]),
            hint_used=bool(r["hint_used"]),
            answered_at=r["answered_at"],
        )
        for r in conn.execute(
            "SELECT * FROM quiz_attempt_items WHERE attempt_id = ? ORDER BY position",
            (attempt.id,),
        )
    ]
    return attempt


def get_attempt(
    conn: sqlite3.Connection, notebook_id: int, attempt_id: int
) -> tuple[Quiz, Attempt]:
    row = conn.execute(
        "SELECT a.* FROM quiz_attempts a JOIN quizzes q ON q.id = a.quiz_id "
        "WHERE a.id = ? AND q.notebook_id = ?",
        (attempt_id, notebook_id),
    ).fetchone()
    if row is None:
        raise AttemptNotFoundError(f"Attempt {attempt_id} not found in this notebook")
    return get_quiz(conn, notebook_id, row["quiz_id"]), _attempt(conn, row)


def list_attempts(conn: sqlite3.Connection, quiz_id: int) -> list[Attempt]:
    rows = conn.execute(
        "SELECT * FROM quiz_attempts WHERE quiz_id = ? ORDER BY id DESC", (quiz_id,)
    ).fetchall()
    return [_attempt(conn, r) for r in rows]


def _create_attempt(
    conn: sqlite3.Connection,
    quiz_id: int,
    question_ids: list[int],
    mode: str,
    parent: int | None,
) -> int:
    cur = conn.execute(
        "INSERT INTO quiz_attempts (quiz_id, mode, parent_attempt_id, total) "
        "VALUES (?, ?, ?, ?)",
        (quiz_id, mode, parent, len(question_ids)),
    )
    attempt_id = cur.lastrowid
    assert attempt_id is not None
    conn.executemany(
        "INSERT INTO quiz_attempt_items (attempt_id, question_id, position) "
        "VALUES (?, ?, ?)",
        [(attempt_id, qid, i) for i, qid in enumerate(question_ids, 1)],
    )
    return attempt_id


def start_attempt(conn: sqlite3.Connection, notebook_id: int, quiz_id: int) -> int:
    """A new full attempt. An unfinished one is abandoned (kept in history)."""
    quiz = get_quiz(conn, notebook_id, quiz_id)
    if quiz.status != "ready" or quiz.question_count == 0:
        raise StudyError("This quiz has no questions to answer")
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM quiz_questions WHERE quiz_id = ? ORDER BY position",
            (quiz_id,),
        )
    ]
    with conn:
        conn.execute(
            "UPDATE quiz_attempts SET status = 'abandoned' WHERE quiz_id = ? "
            "AND status = 'in_progress'",
            (quiz_id,),
        )
        return _create_attempt(conn, quiz_id, ids, "full", None)


def retry_mistakes(conn: sqlite3.Connection, notebook_id: int, attempt_id: int) -> int:
    """A new attempt with only the questions missed (wrong or skipped) in a
    completed attempt. The earlier attempt is not changed."""
    quiz, attempt = get_attempt(conn, notebook_id, attempt_id)
    if attempt.status != "completed":
        raise StudyError("Finish this attempt first")
    missed = [i.question_id for i in attempt.missed]
    if not missed:
        raise StudyError("No mistakes to retry in this attempt")
    with conn:
        conn.execute(
            "UPDATE quiz_attempts SET status = 'abandoned' WHERE quiz_id = ? "
            "AND status = 'in_progress'",
            (quiz.id,),
        )
        return _create_attempt(conn, quiz.id, missed, "mistakes", attempt.id)


def record_hint(
    conn: sqlite3.Connection, notebook_id: int, attempt_id: int, position: int
) -> None:
    _, attempt = get_attempt(conn, notebook_id, attempt_id)
    item = next((i for i in attempt.items if i.position == position), None)
    if item is None:
        raise AttemptNotFoundError("No such question in this attempt")
    if attempt.status != "in_progress" or item.answered:
        return  # viewing the hint after answering changes nothing
    with conn:
        conn.execute(
            "UPDATE quiz_attempt_items SET hint_used = 1 WHERE id = ?", (item.id,)
        )


def answer(
    conn: sqlite3.Connection,
    notebook_id: int,
    attempt_id: int,
    position: int,
    selected: list[int],
) -> Item:
    """Record the answer to one item (once). Completes the attempt when it was
    the last unanswered item. ``selected`` holds 1-based option positions."""
    quiz, attempt = get_attempt(conn, notebook_id, attempt_id)
    item = next((i for i in attempt.items if i.position == position), None)
    if item is None:
        raise AttemptNotFoundError("No such question in this attempt")
    if item.answered:
        raise AlreadyAnsweredError("This question was already answered")
    if attempt.status != "in_progress":
        raise StudyError("This attempt is finished; start a new one to answer again")
    question = questions(conn, quiz.id, [item.question_id])[0]
    valid = {o.position for o in question.options}
    chosen = sorted(set(selected))
    if not chosen or not set(chosen) <= valid:
        raise StudyError("Choose an answer first")
    if question.qtype in ("single", "true_false") and len(chosen) != 1:
        raise StudyError("Choose exactly one answer")
    correct = chosen == question.correct_positions
    with conn:
        updated = conn.execute(
            f"UPDATE quiz_attempt_items SET selected = ?, is_correct = ?, "
            f"answered_at = {NOW_SQL} WHERE id = ? AND answered_at IS NULL",
            (json.dumps(chosen), int(correct), item.id),
        ).rowcount
        if updated != 1:
            raise AlreadyAnsweredError("This question was already answered")
        _apply_stats(conn, notebook_id, question.concepts, correct)
        remaining = conn.execute(
            "SELECT count(*) FROM quiz_attempt_items WHERE attempt_id = ? "
            "AND answered_at IS NULL",
            (attempt_id,),
        ).fetchone()[0]
        if remaining == 0:
            _complete(conn, attempt_id)
    item.selected, item.is_correct = chosen, correct
    return item


def _complete(conn: sqlite3.Connection, attempt_id: int) -> None:
    conn.execute(
        f"UPDATE quiz_attempts SET status = 'completed', completed_at = {NOW_SQL}, "
        "score = (SELECT count(*) FROM quiz_attempt_items WHERE attempt_id = ? "
        "AND is_correct = 1) WHERE id = ? AND status = 'in_progress'",
        (attempt_id, attempt_id),
    )


def finish_attempt(conn: sqlite3.Connection, notebook_id: int, attempt_id: int) -> None:
    """Finish early: unanswered questions count as not correct (skipped)."""
    _, attempt = get_attempt(conn, notebook_id, attempt_id)
    if attempt.status != "in_progress":
        return
    with conn:
        _complete(conn, attempt_id)


# --- study progress ----------------------------------------------------------------


def _apply_stats(
    conn: sqlite3.Connection,
    notebook_id: int,
    concepts: list[dict[str, str]],
    correct: bool,
) -> None:
    for c in concepts:
        key, label = c.get("key", ""), c.get("label", "")
        if not key:
            continue
        conn.execute(
            "INSERT INTO study_concept_stats (notebook_id, concept_key, label) "
            "VALUES (?, ?, ?) ON CONFLICT (notebook_id, concept_key) DO NOTHING",
            (notebook_id, key[:80], label[:80] or key[:80]),
        )
        conn.execute(
            f"""UPDATE study_concept_stats SET
                  times_seen = times_seen + 1,
                  correct_count = correct_count + ?,
                  incorrect_count = incorrect_count + ?,
                  streak = CASE WHEN ?
                    THEN (CASE WHEN streak > 0 THEN streak + 1 ELSE 1 END)
                    ELSE (CASE WHEN streak < 0 THEN streak - 1 ELSE -1 END) END,
                  last_result = ?, last_seen = {NOW_SQL}
                WHERE notebook_id = ? AND concept_key = ?""",
            (
                int(correct),
                int(not correct),
                int(correct),
                "correct" if correct else "incorrect",
                notebook_id,
                key[:80],
            ),
        )


def _rebuild_stats(conn: sqlite3.Connection, notebook_id: int) -> None:
    conn.execute(
        "DELETE FROM study_concept_stats WHERE notebook_id = ?", (notebook_id,)
    )
    for r in derived_stats(conn, notebook_id):
        conn.execute(
            "INSERT INTO study_concept_stats (notebook_id, concept_key, label, "
            "times_seen, correct_count, incorrect_count, streak, last_result, "
            "last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (notebook_id, *r),
        )


def rebuild_concept_stats(conn: sqlite3.Connection, notebook_id: int) -> None:
    with conn:
        _rebuild_stats(conn, notebook_id)


def derived_stats(conn: sqlite3.Connection, notebook_id: int) -> list[tuple[Any, ...]]:
    """(key, label, seen, correct, incorrect, streak, last_result, last_seen)
    computed from every answered item of this notebook, oldest first."""
    stats: dict[str, list[Any]] = {}
    for r in conn.execute(
        "SELECT i.is_correct, i.answered_at, q.concepts FROM quiz_attempt_items i "
        "JOIN quiz_questions q ON q.id = i.question_id "
        "JOIN quizzes z ON z.id = q.quiz_id WHERE z.notebook_id = ? "
        "AND i.answered_at IS NOT NULL ORDER BY i.answered_at, i.id",
        (notebook_id,),
    ):
        ok = bool(r["is_correct"])
        for c in json.loads(r["concepts"] or "[]"):
            key = (c.get("key") or "")[:80]
            if not key:
                continue
            s = stats.setdefault(
                key, [(c.get("label") or key)[:80], 0, 0, 0, 0, None, None]
            )
            s[1] += 1
            s[2 if ok else 3] += 1
            s[4] = (
                (s[4] + 1 if s[4] > 0 else 1) if ok else (s[4] - 1 if s[4] < 0 else -1)
            )
            s[5] = "correct" if ok else "incorrect"
            s[6] = r["answered_at"]
    return [(key, *values) for key, values in sorted(stats.items())]


def stats_consistent(conn: sqlite3.Connection, notebook_id: int) -> bool:
    stored = [
        tuple(r)
        for r in conn.execute(
            "SELECT concept_key, label, times_seen, correct_count, incorrect_count, "
            "streak, last_result FROM study_concept_stats WHERE notebook_id = ? "
            "ORDER BY concept_key",
            (notebook_id,),
        )
    ]
    derived = [r[:7] for r in derived_stats(conn, notebook_id)]
    return stored == derived


@dataclass
class ConceptProgress:
    key: str
    label: str
    times_seen: int
    correct: int
    incorrect: int
    streak: int
    last_result: str | None
    last_seen: str | None

    @property
    def status(self) -> str:
        """Labels from actual attempt data only — never "mastered"."""
        if self.streak < 0:
            return "Needs review"
        if self.streak >= 2:
            return "Strong recently"
        return "Seen"


def concept_progress(
    conn: sqlite3.Connection,
    notebook_id: int,
    *,
    needs_review: bool = False,
    limit: int = 50,
) -> list[ConceptProgress]:
    where = " AND streak < 0" if needs_review else ""
    rows = conn.execute(
        "SELECT concept_key, label, times_seen, correct_count, incorrect_count, "
        "streak, last_result, last_seen FROM study_concept_stats "
        f"WHERE notebook_id = ?{where} "
        "ORDER BY (streak < 0) DESC, last_seen DESC LIMIT ?",
        (notebook_id, limit),
    ).fetchall()
    return [ConceptProgress(*r) for r in rows]


def attempt_concepts_to_review(
    conn: sqlite3.Connection, attempt: Attempt, quiz_id: int
) -> list[str]:
    """Concept labels of the questions missed in this attempt (deduplicated)."""
    missed = {i.question_id for i in attempt.missed}
    labels: list[str] = []
    for q in questions(conn, quiz_id, list(missed)) if missed else []:
        for c in q.concepts:
            if c.get("label") and c["label"] not in labels:
                labels.append(c["label"])
    return labels
