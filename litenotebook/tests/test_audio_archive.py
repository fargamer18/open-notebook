"""Phase 14: Audio Overviews in exports, backups and imports (format 3), and
compatibility with Phase 12 (format 1) and Phase 13 (format 2) archives."""

import json
import threading
from pathlib import Path

import pytest

from archive_fixture import edit_rows, read_entries, write_archive
from audio_fixtures import PDF_TRAP, FakeSpeech, install_fake_voices
from litenotebook.ai import speech
from litenotebook.archive.export import export_notebook
from litenotebook.archive.format import FORMAT_VERSION, ArchiveError, Limits
from litenotebook.archive.importer import import_archive
from litenotebook.archive.reader import verify_archive
from litenotebook.db import available_migrations, open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source
from litenotebook.study import audio_store as store
from litenotebook.study import store as quiz_store
from litenotebook.study.audio_render import SynthJob, run_synthesis
from quiz_fixtures import build_notebook
from test_audio_web import saved_script

FIXTURES = Path(__file__).parent / "fixtures"
CURRENT_SCHEMA = len(available_migrations())


@pytest.fixture
def db(settings):
    c = open_db(settings.db_path)
    yield c
    c.close()


def speak(settings, nb_id, oid):
    install_fake_voices(settings)
    c = open_db(settings.db_path)
    try:
        aid = store.start_asset(c, oid, "fake 1", {"A": "voice-a", "B": "voice-b"})
    finally:
        c.close()
    voices = {
        "A": speech.voice(settings, "voice-a"),
        "B": speech.voice(settings, "voice-b"),
    }
    run_synthesis(
        SynthJob(
            settings, nb_id, oid, aid, voices, FakeSpeech(), cancel=threading.Event()
        )
    )
    return aid


def audio_snapshot(conn, nb_id):
    q = lambda sql: [tuple(r) for r in conn.execute(sql, (nb_id,))]  # noqa: E731
    return {
        "overviews": q(
            "SELECT title, format, target_minutes, focus, status, word_count, "
            "words_per_minute, message, created_at FROM audio_overviews "
            "WHERE notebook_id = ? ORDER BY created_at, id"
        ),
        "turns": q(
            "SELECT o.title, s.position, s.title, t.position, t.speaker, t.text, "
            "t.refs FROM audio_turns t JOIN audio_sections s ON s.id = t.section_id "
            "JOIN audio_overviews o ON o.id = s.overview_id WHERE o.notebook_id = ? "
            "ORDER BY o.id, s.position, t.position"
        ),
        "evidence": q(
            "SELECT o.title, e.label, e.kind, e.title, e.page_no, e.excerpt, "
            "e.source_id IS NULL FROM audio_evidence e JOIN audio_overviews o "
            "ON o.id = e.overview_id WHERE o.notebook_id = ? ORDER BY o.id, e.label"
        ),
        "assets": q(
            "SELECT o.title, a.status, a.engine, a.voices, a.sample_rate, "
            "a.duration_s, a.size_bytes, a.sha256, a.chapters FROM audio_assets a "
            "JOIN audio_overviews o ON o.id = a.overview_id WHERE o.notebook_id = ? "
            "ORDER BY o.id, a.id"
        ),
    }


def build(settings, tmp_path, db):
    nb_id, src = build_notebook(
        settings,
        tmp_path,
        name="Audio",
        sources={
            "Trap.pdf": PDF_TRAP,
            "Gone.pdf": ["Gone page about the Ember stage."],
        },
    )
    kept = saved_script(settings, nb_id, src["Trap.pdf"], title="Kept")
    speak(settings, nb_id, kept)
    gone = saved_script(
        settings,
        nb_id,
        src["Gone.pdf"],
        title="Gone source",
        texts=["The Ember stage is described.", "What else?", "Ember is a stage."],
    )
    delete_source(db, settings, nb_id, src["Gone.pdf"], FaissStore(db, settings))
    failed = store.create_overview(
        db,
        nb_id,
        title="Failed",
        fmt="narrator",
        minutes=5,
        focus="",
        material={},
        wpm=150,
    )
    store.fail(db, failed, "x")
    return nb_id, src, kept, gone


def test_round_trip_preserves_transcripts_evidence_and_audio(db, settings, tmp_path):
    nb_id, src, kept, _ = build(settings, tmp_path, db)
    before = audio_snapshot(db, nb_id)
    result = export_notebook(db, settings, nb_id, tmp_path / "x.zip")
    assert result.manifest["export_format_version"] == FORMAT_VERSION == 3
    assert result.counts["audio_overviews"] == 3 and result.counts["audio_assets"] == 1
    assert result.counts["audio_bytes"] > 1000
    entries = read_entries(result.path)
    wavs = [n for n in entries if n.endswith(".wav")]
    assert len(wavs) == 1 and entries[wavs[0]][:4] == b"RIFF"
    _, results = import_archive(db, settings, result.path)
    new = results[0].notebook_id
    assert results[0].report.ok, results[0].report.problems
    after = audio_snapshot(db, new)
    assert after["overviews"] == before["overviews"]
    assert after["turns"] == before["turns"]
    assert after["assets"] == before["assets"]
    # evidence: the deleted source's snapshot stays, its id was not exported
    assert [e[:6] for e in after["evidence"]] == [e[:6] for e in before["evidence"]]
    new_kept = next(o for o in store.list_overviews(db, new) if o.title == "Kept")
    asset = store.list_assets(db, new_kept.id)[0]
    path = store.asset_path(settings, new, new_kept.id, asset.id)
    assert (
        path.read_bytes()
        == store.asset_path(
            settings, nb_id, kept, store.list_assets(db, kept)[0].id
        ).read_bytes()
    )
    turn_ids = [t.id for s in store.sections(db, new_kept.id) for t in s.turns]
    assert [t[0] for t in asset.turn_times] == turn_ids  # remapped to the new turns
    ev = store.evidence(db, new_kept.id)["S1"]
    new_sources = [
        r[0] for r in db.execute("SELECT id FROM sources WHERE notebook_id = ?", (new,))
    ]
    assert ev.source_id in new_sources and ev.exists and ev.chunk_exists
    gone = next(o for o in store.list_overviews(db, new) if o.title == "Gone source")
    assert all(not e.exists for e in store.evidence(db, gone.id).values())
    assert results[0].report.counts["audio_files"] == 1


def test_previous_archive_versions_still_import(db, settings):
    for name, version, quizzes in (
        ("phase12_notebook_v1.zip", 1, 0),
        ("phase13_notebook_v2.zip", 2, 1),
    ):
        report = verify_archive(
            FIXTURES / name,
            Limits.from_settings(settings),
            current_schema=CURRENT_SCHEMA,
        )
        assert report.manifest["export_format_version"] == version
        _, results = import_archive(db, settings, FIXTURES / name)
        nb = results[0]
        assert nb.report.ok, nb.report.problems
        assert nb.report.counts["audio_overviews"] == 0
        assert nb.report.counts["quizzes"] == quizzes
        if quizzes:
            quiz = quiz_store.list_quizzes(db, nb.notebook_id)[0]
            assert quiz.title == "Fixture quiz" and quiz.attempts == 1


def test_corrupt_or_mismatched_audio_imports_nothing(db, settings, tmp_path):
    nb_id, *_ = build(settings, tmp_path, db)
    entries = read_entries(
        export_notebook(db, settings, nb_id, tmp_path / "x.zip").path
    )
    wav = next(n for n in entries if n.endswith(".wav"))
    cases = []
    tampered = dict(entries)
    tampered[wav] = b"NOTAWAVE" + entries[wav][8:]
    cases.append((tampered, "size or SHA-256"))
    missing = dict(entries)
    del missing[wav]
    cases.append((missing, "do not match"))
    extra = dict(entries)
    extra[wav.replace(wav.rsplit("/", 1)[1], "999.wav")] = entries[wav]
    cases.append((extra, "do not match"))
    old = dict(entries)
    manifest = json.loads(old["manifest.json"])
    manifest["export_format_version"] = 2
    old["manifest.json"] = json.dumps(manifest).encode()
    cases.append((old, "Unknown table"))
    unfinished = dict(entries)
    edit_rows(
        unfinished,
        f"notebooks/{nb_id}/tables/audio_assets.jsonl",
        lambda rows: rows[0].update(status="synthesizing"),
    )
    cases.append((unfinished, "only finished audio"))
    count = db.execute("SELECT count(*) FROM audio_overviews").fetchone()[0]
    for i, (case, match) in enumerate(cases):
        bad = write_archive(tmp_path / f"bad{i}.zip", case)
        with pytest.raises(ArchiveError, match=match):
            import_archive(db, settings, bad)
        assert db.execute("SELECT count(*) FROM audio_overviews").fetchone()[0] == count


def test_missing_audio_file_stops_the_export(db, settings, tmp_path):
    nb_id, _, kept, _ = build(settings, tmp_path, db)
    aid = store.list_assets(db, kept)[0].id
    store.asset_path(settings, nb_id, kept, aid).unlink()
    with pytest.raises(ArchiveError, match="audio file for “Kept” is missing"):
        export_notebook(db, settings, nb_id, tmp_path / "x.zip")
