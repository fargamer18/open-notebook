"""Phase 13: quiz schema, immutability, attempts, scoring, retry and study stats."""

import sqlite3

import pytest

from litenotebook.db import schema_version
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source
from litenotebook.notebooks import create_notebook, delete_notebook
from litenotebook.notes import create_note, delete_note
from litenotebook.study import store
from litenotebook.study.model import Concept, EvidenceItem, QuizQuestion
from quiz_fixtures import (
    PDF_A,
    build_notebook,
    correct_choice,
    make_ready_quiz,
    wrong_choice,
)


@pytest.fixture
def nb(settings, tmp_path):
    return build_notebook(settings, tmp_path, sources={"Atlas.pdf": PDF_A})


@pytest.fixture
def db(settings, nb):
    from litenotebook.db import open_db

    c = open_db(settings.db_path)
    yield c
    c.close()


def test_schema_has_study_tables(db):
    from litenotebook.db import available_migrations

    assert schema_version(db) == len(available_migrations()) >= 9
    tables = {
        r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "quizzes",
        "quiz_questions",
        "quiz_question_options",
        "quiz_evidence",
        "quiz_attempts",
        "quiz_attempt_items",
        "study_concept_stats",
    } <= tables


def test_quiz_crud_with_options_and_evidence(db, nb):
    nb_id, sources = nb
    quiz_id = make_ready_quiz(db, nb_id, sources["Atlas.pdf"])
    quiz = store.get_quiz(db, nb_id, quiz_id)
    assert quiz.status == "ready" and quiz.question_count == 3
    assert quiz.source_names == ["Atlas.pdf"] and quiz.difficulty_label == "Medium"
    qs = store.questions(db, quiz_id)
    assert [q.qtype for q in qs] == ["single", "multiple", "true_false"]
    assert [q.position for q in qs] == [1, 2, 3]
    assert len(qs[0].options) == 4 and sum(o.is_correct for o in qs[0].options) == 1
    assert sum(o.is_correct for o in qs[1].options) == 2
    assert [o.text for o in qs[2].options] == ["True", "False"]
    ev = qs[0].evidence[0]
    assert ev.kind == "source" and ev.exists and ev.chunk_exists and ev.page_no == 1
    assert ev.tag == "S1" and ev.label == "Atlas.pdf" and "Quillport" in ev.excerpt
    # another notebook cannot see it
    other = create_notebook(db, "Other")
    with pytest.raises(store.QuizNotFoundError):
        store.get_quiz(db, other.id, quiz_id)
    assert [q.id for q in store.list_quizzes(db, nb_id)] == [quiz_id]


def test_options_are_shuffled_but_correctness_preserved(db, nb):
    nb_id, _ = nb
    found = []
    for i in range(12):
        quiz_id = make_ready_quiz(db, nb_id, title=f"Q{i}")
        q = store.questions(db, quiz_id)[0]
        correct = [o for o in q.options if o.is_correct]
        assert [o.text for o in correct] == ["Quillport"]
        found.append(correct[0].position)
    assert len(set(found)) > 1  # the answer is not always in the same slot


def test_generated_content_is_immutable(db, nb):
    nb_id, sources = nb
    quiz_id = make_ready_quiz(db, nb_id, sources["Atlas.pdf"])
    for sql in (
        "UPDATE quiz_questions SET question = 'changed'",
        "UPDATE quiz_question_options SET is_correct = 1",
        "UPDATE quiz_evidence SET excerpt = 'changed'",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(sql)
    db.rollback()
    assert store.questions(db, quiz_id)[0].question.startswith("Which city")


def test_full_attempt_scoring_hints_and_answer_once(db, nb):
    nb_id, sources = nb
    quiz_id = make_ready_quiz(db, nb_id, sources["Atlas.pdf"])
    aid = store.start_attempt(db, nb_id, quiz_id)
    store.record_hint(db, nb_id, aid, 1)
    store.answer(db, nb_id, aid, 1, correct_choice(db, aid, 1))  # single: right
    with pytest.raises(store.AlreadyAnsweredError):
        store.answer(db, nb_id, aid, 1, wrong_choice(db, aid, 1))
    with pytest.raises(sqlite3.IntegrityError, match="already answered"):
        db.execute(
            "UPDATE quiz_attempt_items SET is_correct = 0 WHERE attempt_id = ?", (aid,)
        )
    db.rollback()
    with pytest.raises(store.StudyError):
        store.answer(db, nb_id, aid, 2, [])  # nothing chosen
    with pytest.raises(store.StudyError):
        store.answer(db, nb_id, aid, 2, [9])  # not an option
    multi = correct_choice(db, aid, 2)
    store.answer(db, nb_id, aid, 2, multi[:1])  # multi: partially right = wrong
    _, attempt = store.get_attempt(db, nb_id, aid)
    assert attempt.status == "in_progress" and attempt.answered == 2
    with pytest.raises(store.StudyError):
        store.answer(db, nb_id, aid, 3, [1, 2])  # true/false takes one answer
    store.answer(db, nb_id, aid, 3, correct_choice(db, aid, 3))
    _, attempt = store.get_attempt(db, nb_id, aid)
    assert attempt.status == "completed" and attempt.score == 2 and attempt.total == 3
    assert attempt.items[0].hint_used and not attempt.items[1].hint_used
    assert [i.is_correct for i in attempt.items] == [True, False, True]
    with pytest.raises(store.StudyError):
        store.answer(db, nb_id, aid, 3, [1])


def test_incomplete_attempt_resume_finish_early_and_retake(db, nb):
    nb_id, sources = nb
    quiz_id = make_ready_quiz(db, nb_id, sources["Atlas.pdf"])
    aid = store.start_attempt(db, nb_id, quiz_id)
    store.answer(db, nb_id, aid, 1, correct_choice(db, aid, 1))
    quiz = store.get_quiz(db, nb_id, quiz_id)
    assert quiz.open_attempt == aid  # resumable
    store.finish_attempt(db, nb_id, aid)
    _, attempt = store.get_attempt(db, nb_id, aid)
    assert attempt.status == "completed" and attempt.score == 1 and attempt.skipped == 2
    # retake: a new attempt, the old one untouched
    again = store.start_attempt(db, nb_id, quiz_id)
    open_again = store.start_attempt(db, nb_id, quiz_id)  # abandons the open one
    statuses = dict(db.execute("SELECT id, status FROM quiz_attempts").fetchall())
    assert statuses == {aid: "completed", again: "abandoned", open_again: "in_progress"}
    quiz = store.get_quiz(db, nb_id, quiz_id)
    assert quiz.attempts == 1 and quiz.best == 1 and quiz.latest == (1, 3)


def test_retry_mistakes_is_a_new_attempt_of_missed_questions(db, nb):
    nb_id, sources = nb
    quiz_id = make_ready_quiz(db, nb_id, sources["Atlas.pdf"])
    aid = store.start_attempt(db, nb_id, quiz_id)
    store.answer(db, nb_id, aid, 1, wrong_choice(db, aid, 1))
    store.answer(db, nb_id, aid, 2, correct_choice(db, aid, 2))
    with pytest.raises(store.StudyError):
        store.retry_mistakes(db, nb_id, aid)  # not finished yet
    store.finish_attempt(db, nb_id, aid)  # question 3 skipped
    retry = store.retry_mistakes(db, nb_id, aid)
    _, first = store.get_attempt(db, nb_id, aid)
    _, second = store.get_attempt(db, nb_id, retry)
    assert second.mode == "mistakes" and second.parent_attempt_id == aid
    assert [i.question_id for i in second.items] == [
        first.items[0].question_id,
        first.items[2].question_id,
    ]
    store.answer(db, nb_id, retry, 1, correct_choice(db, retry, 1))
    store.answer(db, nb_id, retry, 2, correct_choice(db, retry, 2))
    _, second = store.get_attempt(db, nb_id, retry)
    _, first_again = store.get_attempt(db, nb_id, aid)
    assert second.score == 2 and first_again.score == 1  # history kept
    assert first_again.items[0].is_correct is False
    with pytest.raises(store.StudyError):
        store.retry_mistakes(db, nb_id, retry)  # no mistakes left


def test_concept_stats_streaks_and_labels(db, nb):
    nb_id, sources = nb
    quiz_id = make_ready_quiz(db, nb_id, sources["Atlas.pdf"])
    for answers in (("wrong",), ("right",), ("right",)):
        aid = store.start_attempt(db, nb_id, quiz_id)
        pick = wrong_choice if answers[0] == "wrong" else correct_choice
        store.answer(db, nb_id, aid, 1, pick(db, aid, 1))
        if answers[0] == "wrong":
            progress = {c.key: c for c in store.concept_progress(db, nb_id)}
            assert progress["zorbania capital"].status == "Needs review"
            assert [
                c.key for c in store.concept_progress(db, nb_id, needs_review=True)
            ] == ["zorbania capital"]
    progress = {c.key: c for c in store.concept_progress(db, nb_id)}
    c = progress["zorbania capital"]
    assert (c.times_seen, c.correct, c.incorrect, c.streak) == (3, 2, 1, 2)
    assert c.status == "Strong recently"  # never "mastered"
    assert store.stats_consistent(db, nb_id)
    db.execute("UPDATE study_concept_stats SET correct_count = 99")
    db.commit()
    assert not store.stats_consistent(db, nb_id)
    store.rebuild_concept_stats(db, nb_id)
    assert store.stats_consistent(db, nb_id)


def test_delete_quiz_removes_history_and_recomputes_stats(db, nb):
    nb_id, sources = nb
    keep = make_ready_quiz(db, nb_id, sources["Atlas.pdf"], title="Keep")
    gone = make_ready_quiz(db, nb_id, sources["Atlas.pdf"], title="Gone")
    for quiz_id in (keep, gone):
        aid = store.start_attempt(db, nb_id, quiz_id)
        store.answer(db, nb_id, aid, 1, correct_choice(db, aid, 1))
    assert store.concept_progress(db, nb_id)[0].times_seen == 2
    store.delete_quiz(db, nb_id, gone)
    assert [q.title for q in store.list_quizzes(db, nb_id)] == ["Keep"]
    for table in (
        "quiz_questions",
        "quiz_attempts",
        "quiz_attempt_items",
        "quiz_evidence",
    ):
        left = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert left > 0  # the other quiz's rows stay
    assert (
        db.execute(
            "SELECT count(*) FROM quiz_questions WHERE quiz_id = ?", (gone,)
        ).fetchone()[0]
        == 0
    )
    assert store.concept_progress(db, nb_id)[0].times_seen == 1
    assert store.stats_consistent(db, nb_id)


def test_deleted_source_keeps_evidence_snapshot(db, nb, settings):
    nb_id, sources = nb
    quiz_id = make_ready_quiz(db, nb_id, sources["Atlas.pdf"])
    delete_source(db, settings, nb_id, sources["Atlas.pdf"], FaissStore(db, settings))
    ev = store.questions(db, quiz_id)[0].evidence[0]
    assert not ev.exists and not ev.chunk_exists
    assert "Quillport" in ev.excerpt and ev.page_no == 1 and ev.label == "Atlas.pdf"


def test_note_and_chat_evidence_snapshots(db, nb):
    nb_id, _ = nb
    note = create_note(db, nb_id, "Harbour notes", "Quillport has a deep harbour.")
    quiz_id = store.create_quiz(
        db,
        nb_id,
        title="N",
        difficulty="easy",
        types=["true_false"],
        count=5,
        focus="",
        material={"sources": [], "notes": True, "chat": False},
    )
    ev = EvidenceItem(
        "N1", "note", "Quillport has a deep harbour.", "Harbour notes", note_id=note.id
    )
    q = QuizQuestion(
        "true_false",
        "Quillport has a deep harbour.",
        ["True", "False"],
        [0],
        "The note says so.",
        "",
        [ev],
        [Concept("harbour", "Harbour")],
    )
    store.save_questions(db, quiz_id, [q], message=None, model="m", meta={})
    saved = store.questions(db, quiz_id)[0].evidence[0]
    assert saved.kind == "note" and saved.tag == "N1" and saved.exists
    delete_note(db, nb_id, note.id)
    saved = store.questions(db, quiz_id)[0].evidence[0]
    assert not saved.exists and saved.excerpt == "Quillport has a deep harbour."


def test_recover_generating_and_notebook_delete_cascade(db, nb, settings):
    nb_id, sources = nb
    pending = store.create_quiz(
        db,
        nb_id,
        title="P",
        difficulty="easy",
        types=["single"],
        count=5,
        focus="",
        material={},
    )
    assert store.recover_generating(db) == 1
    quiz = store.get_quiz(db, nb_id, pending)
    assert quiz.status == "failed" and "interrupted" in quiz.error
    make_ready_quiz(db, nb_id, sources["Atlas.pdf"])
    delete_notebook(db, settings, nb_id)
    for table in ("quizzes", "quiz_questions", "quiz_evidence", "study_concept_stats"):
        assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
