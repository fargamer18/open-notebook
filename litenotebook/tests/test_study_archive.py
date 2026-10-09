"""Phase 13: quizzes in notebook exports, backups and imports (format version 2),
and compatibility with Phase 10–12 archives (format version 1)."""

import json
from pathlib import Path

import pytest

from archive_fixture import edit_rows, read_entries, write_archive
from litenotebook.archive.export import create_backup, export_notebook
from litenotebook.archive.format import (
    FORMAT_VERSION,
    TABLE_SINCE,
    ArchiveError,
    Limits,
)
from litenotebook.archive.importer import import_archive
from litenotebook.archive.integrity import notebook_report
from litenotebook.archive.reader import verify_archive
from litenotebook.db import available_migrations, open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source
from litenotebook.notes import create_note
from litenotebook.study import store
from litenotebook.study.model import Concept, EvidenceItem, QuizQuestion
from quiz_fixtures import (
    PDF_A,
    PDF_B,
    build_notebook,
    correct_choice,
    make_ready_quiz,
    wrong_choice,
)

FIXTURE_V1 = Path(__file__).parent / "fixtures" / "phase12_notebook_v1.zip"
CURRENT_SCHEMA = len(available_migrations())


@pytest.fixture
def db(settings):
    c = open_db(settings.db_path)
    yield c
    c.close()


def study_snapshot(conn, nb_id):
    """User-visible quiz content without ids."""
    q = lambda sql: [tuple(r) for r in conn.execute(sql, (nb_id,))]  # noqa: E731
    return {
        "quizzes": q(
            "SELECT title, status, difficulty, question_types, requested_count, "
            "question_count, focus, message, created_at FROM quizzes "
            "WHERE notebook_id = ? ORDER BY created_at, title"
        ),
        "questions": q(
            "SELECT z.title, q.position, q.qtype, q.question, q.explanation, q.hint, "
            "q.concepts FROM quiz_questions q JOIN quizzes z ON z.id = q.quiz_id "
            "WHERE z.notebook_id = ? ORDER BY z.title, q.position"
        ),
        "options": q(
            "SELECT z.title, q.position, o.position, o.text, o.is_correct FROM "
            "quiz_question_options o JOIN quiz_questions q ON q.id = o.question_id "
            "JOIN quizzes z ON z.id = q.quiz_id WHERE z.notebook_id = ? "
            "ORDER BY z.title, q.position, o.position"
        ),
        "evidence": q(
            "SELECT z.title, q.position, e.kind, e.label, e.page_no, e.excerpt, "
            "CASE e.kind WHEN 'source' THEN EXISTS (SELECT 1 FROM chunks c WHERE "
            "c.id = e.chunk_id AND c.source_id = e.source_id AND c.text LIKE "
            "'%' || substr(e.excerpt, 1, 20) || '%') "
            "WHEN 'note' THEN EXISTS (SELECT 1 FROM notes n WHERE n.id = e.note_id "
            "AND n.notebook_id = z.notebook_id) ELSE 0 END "
            "FROM quiz_evidence e JOIN quiz_questions q ON q.id = e.question_id "
            "JOIN quizzes z ON z.id = q.quiz_id WHERE z.notebook_id = ? "
            "ORDER BY z.title, q.position, e.ordinal"
        ),
        "attempts": q(
            "SELECT z.title, a.mode, a.status, a.total, a.score, a.started_at, "
            "a.parent_attempt_id IS NOT NULL FROM quiz_attempts a JOIN quizzes z "
            "ON z.id = a.quiz_id WHERE z.notebook_id = ? ORDER BY a.started_at, a.mode"
        ),
        "answers": q(
            "SELECT z.title, a.mode, i.position, i.selected, i.is_correct, "
            "i.hint_used, i.answered_at, q.position FROM quiz_attempt_items i "
            "JOIN quiz_attempts a ON a.id = i.attempt_id JOIN quizzes z ON "
            "z.id = a.quiz_id JOIN quiz_questions q ON q.id = i.question_id "
            "WHERE z.notebook_id = ? ORDER BY a.started_at, a.mode, i.position"
        ),
        "stats": q(
            "SELECT concept_key, label, times_seen, correct_count, incorrect_count, "
            "streak, last_result, last_seen FROM study_concept_stats "
            "WHERE notebook_id = ? ORDER BY concept_key"
        ),
    }


def study_notebook(db, settings, tmp_path):
    nb_id, src = build_notebook(
        settings,
        tmp_path,
        name="Quizzed",
        sources={"Atlas.pdf": PDF_A, "Gone.pdf": PDF_B},
    )
    note = create_note(db, nb_id, "Harbour", "Quillport has a deep natural harbour.")
    quiz_id = make_ready_quiz(db, nb_id, src["Atlas.pdf"], title="Atlas quiz")
    aid = store.start_attempt(db, nb_id, quiz_id)
    store.record_hint(db, nb_id, aid, 1)
    store.answer(db, nb_id, aid, 1, wrong_choice(db, aid, 1))
    store.answer(db, nb_id, aid, 2, correct_choice(db, aid, 2))
    store.finish_attempt(db, nb_id, aid)
    retry = store.retry_mistakes(db, nb_id, aid)
    store.answer(db, nb_id, retry, 1, correct_choice(db, retry, 1))
    # a second quiz whose source is deleted and which cites a note
    gone_chunk = db.execute(
        "SELECT id, page_no, text FROM chunks WHERE source_id = ?", (src["Gone.pdf"],)
    ).fetchone()
    other = store.create_quiz(
        db,
        nb_id,
        title="Mixed quiz",
        difficulty="exam",
        types=["true_false"],
        count=5,
        focus="matrices",
        material={
            "sources": [{"id": src["Gone.pdf"], "filename": "Gone.pdf"}],
            "notes": True,
            "chat": False,
        },
    )
    store.save_questions(
        db,
        other,
        [
            QuizQuestion(
                "true_false",
                "A diagonalisable matrix has independent eigenvectors.",
                ["True", "False"],
                [0],
                "Stated in the notes.",
                "",
                [
                    EvidenceItem(
                        "S1",
                        "source",
                        gone_chunk["text"],
                        "Gone.pdf",
                        source_id=src["Gone.pdf"],
                        chunk_id=gone_chunk["id"],
                        page_no=gone_chunk["page_no"],
                        chunk_ord=0,
                    ),
                    EvidenceItem(
                        "N1",
                        "note",
                        "Quillport has a deep natural harbour.",
                        "Harbour",
                        note_id=note.id,
                    ),
                ],
                [Concept("diagonalisation", "Diagonalisation")],
            ),
        ],
        message="Generated 1 of 5 requested questions.",
        model="qwen",
        meta={"x": 1},
    )
    delete_source(db, settings, nb_id, src["Gone.pdf"], FaissStore(db, settings))
    return nb_id, src


def test_export_contains_quiz_tables(db, settings, tmp_path):
    nb_id, _ = study_notebook(db, settings, tmp_path)
    result = export_notebook(db, settings, nb_id, tmp_path / "x.zip")
    assert result.manifest["export_format_version"] == FORMAT_VERSION >= 2
    assert result.manifest["schema_version"] == CURRENT_SCHEMA
    entries = read_entries(result.path)
    for table in TABLE_SINCE:
        assert f"notebooks/{nb_id}/tables/{table}.jsonl" in entries
    counts = result.counts
    assert counts["quizzes"] == 2 and counts["quiz_attempts"] == 2
    assert counts["quiz_evidence"] == 5 and counts["quiz_attempt_items"] == 5


def test_round_trip_preserves_quiz_history_with_new_ids(db, settings, tmp_path):
    nb_id, src = study_notebook(db, settings, tmp_path)
    before = study_snapshot(db, nb_id)
    out = export_notebook(db, settings, nb_id, tmp_path / "x.zip").path
    # make the target's ids differ from the source's
    filler = make_ready_quiz(db, nb_id, None, title="Filler")
    store.start_attempt(db, nb_id, filler)
    _, results = import_archive(db, settings, out)
    new_id = results[0].notebook_id
    assert new_id != nb_id and results[0].report.ok, results[0].report.problems
    after = study_snapshot(db, new_id)
    assert after == before
    # ids were remapped into the new notebook
    new_quiz = next(
        q for q in store.list_quizzes(db, new_id) if q.title == "Atlas quiz"
    )
    old_quiz = next(q for q in store.list_quizzes(db, nb_id) if q.title == "Atlas quiz")
    assert new_quiz.id != old_quiz.id
    new_sources = [
        r[0]
        for r in db.execute("SELECT id FROM sources WHERE notebook_id = ?", (new_id,))
    ]
    assert new_quiz.material["sources"][0]["id"] in new_sources
    ev = store.questions(db, new_quiz.id)[0].evidence[0]
    assert ev.source_id in new_sources and ev.exists and ev.chunk_exists
    mixed = next(q for q in store.list_quizzes(db, new_id) if q.title == "Mixed quiz")
    ev_source, ev_note = store.questions(db, mixed.id)[0].evidence
    assert ev_source.source_id is None and not ev_source.exists  # deleted before export
    assert "diagonalisable" in ev_source.excerpt
    assert ev_note.exists and ev_note.note_id in [
        r[0]
        for r in db.execute("SELECT id FROM notes WHERE notebook_id = ?", (new_id,))
    ]
    assert mixed.material["sources"][0]["id"] is None
    attempts = store.list_attempts(db, new_quiz.id)
    retry = next(a for a in attempts if a.mode == "mistakes")
    assert retry.parent_attempt_id in {a.id for a in attempts}
    assert store.stats_consistent(db, new_id)
    assert notebook_report(db, settings, new_id).counts["quiz_answers"] == 3
    # the imported quiz can be taken again
    aid = store.start_attempt(db, new_id, new_quiz.id)
    store.answer(db, new_id, aid, 1, correct_choice(db, aid, 1))


def test_full_backup_round_trip_includes_study(db, settings, tmp_path):
    nb_id, _ = study_notebook(db, settings, tmp_path)
    before = study_snapshot(db, nb_id)
    backup = create_backup(db, settings, tmp_path / "b.zip").path
    target = open_db(tmp_path / "other" / "db.sqlite3")
    try:
        from dataclasses import replace

        other = replace(settings, data_dir=tmp_path / "other")
        _, results = import_archive(target, other, backup)
        assert study_snapshot(target, results[0].notebook_id) == before
    finally:
        target.close()


def test_phase12_archive_v1_still_imports(db, settings):
    report = verify_archive(
        FIXTURE_V1, Limits.from_settings(settings), current_schema=CURRENT_SCHEMA
    )
    assert report.manifest["export_format_version"] == 1
    assert report.manifest["schema_version"] == 8
    _, results = import_archive(db, settings, FIXTURE_V1)
    nb = results[0]
    assert nb.report.ok, nb.report.problems
    assert nb.report.counts["sources"] == 2 and nb.report.counts["quizzes"] == 0
    assert store.list_quizzes(db, nb.notebook_id) == []
    assert db.execute("SELECT count(*) FROM messages").fetchone()[0] == 4


def test_v1_archive_with_quiz_tables_is_rejected(db, settings, tmp_path):
    nb_id, _ = study_notebook(db, settings, tmp_path)
    entries = read_entries(
        export_notebook(db, settings, nb_id, tmp_path / "x.zip").path
    )
    manifest = json.loads(entries["manifest.json"])
    manifest["export_format_version"] = 1
    entries["manifest.json"] = json.dumps(manifest).encode()
    bad = write_archive(tmp_path / "v1-with-quiz.zip", entries)
    with pytest.raises(ArchiveError, match="Unknown table"):
        verify_archive(
            bad, Limits.from_settings(settings), current_schema=CURRENT_SCHEMA
        )


def test_v2_archive_missing_a_quiz_table_is_rejected(db, settings, tmp_path):
    nb_id, _ = study_notebook(db, settings, tmp_path)
    entries = read_entries(
        export_notebook(db, settings, nb_id, tmp_path / "x.zip").path
    )
    del entries[f"notebooks/{nb_id}/tables/quiz_evidence.jsonl"]
    bad = write_archive(tmp_path / "missing.zip", entries)
    with pytest.raises(ArchiveError, match="quiz_evidence.jsonl is missing"):
        verify_archive(
            bad, Limits.from_settings(settings), current_schema=CURRENT_SCHEMA
        )


@pytest.mark.parametrize(
    ("table", "edit", "match"),
    [
        (
            "quiz_questions",
            lambda rows: rows[0].update(quiz_id=999),
            "not in the archive",
        ),
        (
            "quiz_attempt_items",
            lambda rows: rows[0].update(question_id=999),
            "not in the archive",
        ),
        ("quizzes", lambda rows: rows[0].update(material="{broken"), "Invalid data"),
        ("quizzes", lambda rows: rows[0].update(status="hacked"), "CHECK constraint"),
        ("quiz_attempts", lambda rows: rows[0].update(extra=1), "unexpected columns"),
    ],
)
def test_invalid_quiz_rows_import_nothing(db, settings, tmp_path, table, edit, match):
    nb_id, _ = study_notebook(db, settings, tmp_path)
    entries = read_entries(
        export_notebook(db, settings, nb_id, tmp_path / "x.zip").path
    )
    edit_rows(entries, f"notebooks/{nb_id}/tables/{table}.jsonl", edit)
    bad = write_archive(tmp_path / "bad.zip", entries)
    notebooks = db.execute("SELECT count(*) FROM notebooks").fetchone()[0]
    quizzes = db.execute("SELECT count(*) FROM quizzes").fetchone()[0]
    with pytest.raises(ArchiveError, match=match):
        import_archive(db, settings, bad)
    assert db.execute("SELECT count(*) FROM notebooks").fetchone()[0] == notebooks
    assert db.execute("SELECT count(*) FROM quizzes").fetchone()[0] == quizzes


def test_answers_pointing_at_another_quiz_are_rejected(db, settings, tmp_path):
    nb_id, _ = study_notebook(db, settings, tmp_path)
    entries = read_entries(
        export_notebook(db, settings, nb_id, tmp_path / "x.zip").path
    )
    name = f"notebooks/{nb_id}/tables/quiz_questions.jsonl"
    other_q = [json.loads(line) for line in entries[name].decode().splitlines()][-1][
        "id"
    ]
    edit_rows(
        entries,
        f"notebooks/{nb_id}/tables/quiz_attempt_items.jsonl",
        lambda rows: rows[0].update(question_id=other_q),
    )
    bad = write_archive(tmp_path / "cross.zip", entries)
    with pytest.raises(ArchiveError, match="another quiz"):
        import_archive(db, settings, bad)
