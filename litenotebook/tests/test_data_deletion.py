"""What deleting removes — and what is intentionally kept (Phase 16).

Each delete is checked at three levels: rows, files on disk, and the raw
bytes of the database file (SQLite ``secure_delete`` + FTS5 ``secure-delete``
mean deleted text does not linger in free pages or index segments).

Intentionally retained snapshots, documented in docs/security.md
("Deleting data"): when a *source* is deleted, short excerpts that other
objects cited stay inside those objects — chat citations, quiz evidence,
audio evidence and note links — so history keeps its meaning. Deleting the
quiz, overview, conversation, note or the whole notebook removes them.
"""

from __future__ import annotations

from pathlib import Path

from archive_fixture import build_rich_notebook
from conftest import make_pdf
from litenotebook.archive.importer import import_archive
from litenotebook.chat import complete_message, delete_conversation, start_exchange
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source, ingest_pdf
from litenotebook.notebooks import create_notebook, delete_notebook
from litenotebook.notes import create_note, delete_note, link_chunk
from litenotebook.reader.render import open_source, render_page
from litenotebook.study import audio_store
from litenotebook.study.store import delete_quiz
from quiz_fixtures import make_ready_quiz

FIXTURES = Path(__file__).parent / "fixtures"


def _db_bytes(conn, settings) -> bytes:  # type: ignore[no-untyped-def]
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return b"".join(
        p.read_bytes() for p in settings.db_path.parent.glob("litenotebook.sqlite3*")
    )


def _gone(conn, settings, *tokens: str) -> None:  # type: ignore[no-untyped-def]
    data = _db_bytes(conn, settings).lower()
    for token in tokens:
        assert token.lower().encode() not in data, token


def _count(conn, sql: str, *args) -> int:  # type: ignore[no-untyped-def]
    return int(conn.execute(sql, args).fetchone()[0])


def test_deleting_a_source_removes_its_text_files_vectors_and_cache(
    conn, settings, embedder, store, tmp_path
):
    nb = create_notebook(conn, "Deletion")
    pdf = make_pdf(
        tmp_path / "private.pdf",
        ["Alphacited glacier facts for the quiz.", "Omegauncited diary entry text."],
    )
    report = ingest_pdf(conn, settings, nb, pdf, embedder, store)
    sid = report.source_id
    reader = open_source(conn, settings, nb.id, sid)
    render_page(settings, reader, 1, 1.0)
    quiz = make_ready_quiz(conn, nb.id, sid)  # cites page 1 (snapshot)
    file = settings.source_file(nb.id, sid)
    cache = settings.notebook_dir(nb.id) / "cache" / "pages" / str(sid)
    assert file.is_file() and cache.is_dir()
    vectors_before = (
        FaissStore(conn, settings).count(nb.id)
        if hasattr(FaissStore, "count")
        else None
    )

    delete_source(conn, settings, nb.id, sid, store)

    assert not file.exists() and not cache.exists()
    for table in ("pages", "chunks"):
        assert (
            _count(conn, f"SELECT count(*) FROM {table} WHERE source_id = ?", sid) == 0
        )
    assert (
        _count(
            conn,
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'omegauncited'",
        )
        == 0
    )
    if vectors_before is not None:
        assert FaissStore(conn, settings).count(nb.id) == 0
    _gone(conn, settings, "Omegauncited")
    # documented snapshot: the quiz keeps the excerpt it cited
    assert (
        _count(
            conn,
            "SELECT count(*) FROM quiz_evidence e "
            "JOIN quiz_questions q ON q.id = e.question_id "
            "WHERE q.quiz_id = ? AND e.excerpt LIKE '%Alphacited%'",
            quiz,
        )
        >= 1
    )
    delete_quiz(conn, nb.id, quiz)
    _gone(conn, settings, "Alphacited")


def test_deleting_a_conversation_removes_messages_and_citations(
    conn, settings, tmp_path
):
    rich = build_rich_notebook(conn, settings, tmp_path)
    ex = start_exchange(conn, rich.notebook_id, None, "Zebracrossing question words?")
    complete_message(
        conn, ex.assistant_message_id, "Zebracrossing answer words.", model="q"
    )
    conv = ex.conversation.id
    delete_conversation(conn, rich.notebook_id, conv)
    assert (
        _count(conn, "SELECT count(*) FROM messages WHERE conversation_id = ?", conv)
        == 0
    )
    _gone(conn, settings, "Zebracrossing")


def test_deleting_a_note_removes_its_body_and_links(conn, settings, tmp_path):
    rich = build_rich_notebook(conn, settings, tmp_path)
    note = create_note(
        conn, rich.notebook_id, "Quokkatitle", "Quokkabody private words"
    )
    chunk = conn.execute(
        "SELECT id FROM chunks WHERE notebook_id = ? LIMIT 1", (rich.notebook_id,)
    ).fetchone()[0]
    link_chunk(conn, rich.notebook_id, note.id, chunk)
    delete_note(conn, rich.notebook_id, note.id)
    assert (
        _count(conn, "SELECT count(*) FROM note_links WHERE note_id = ?", note.id) == 0
    )
    _gone(conn, settings, "Quokkabody", "Quokkatitle")


def test_deleting_an_audio_overview_removes_transcript_and_wav(conn, settings):
    _, results = import_archive(conn, settings, FIXTURES / "phase14_notebook_v3.zip")
    nb = results[0].notebook_id
    oid = conn.execute(
        "SELECT id FROM audio_overviews WHERE notebook_id = ?", (nb,)
    ).fetchone()[0]
    turns = [
        r[0]
        for r in conn.execute(
            "SELECT t.text FROM audio_turns t "
            "JOIN audio_sections s ON s.id = t.section_id "
            "WHERE s.overview_id = ?",
            (oid,),
        )
    ]
    wavs = list(settings.notebook_dir(nb).rglob("*.wav"))
    assert turns and wavs
    audio_store.delete_overview(conn, settings, nb, oid)
    assert not any(w.exists() for w in wavs)
    for table in ("audio_sections", "audio_evidence", "audio_assets"):
        assert (
            _count(conn, f"SELECT count(*) FROM {table} WHERE overview_id = ?", oid)
            == 0
        )
    # (turn text is grounded in the source, so it legitimately remains in the
    # source's chunks; rows and files are what the overview owns)


def test_deleting_a_notebook_removes_everything(conn, settings, tmp_path):
    rich = build_rich_notebook(conn, settings, tmp_path)
    create_note(conn, rich.notebook_id, "Narwhaltitle", "Narwhalbody words")
    pdf = make_pdf(tmp_path / "extra.pdf", ["Okapiunique source sentence here."])
    from conftest import FakeEmbedder
    from litenotebook.notebooks import get_notebook

    rich_nb = get_notebook(conn, rich.notebook_id)
    ingest_pdf(conn, settings, rich_nb, pdf, FakeEmbedder(), FaissStore(conn, settings))
    make_ready_quiz(conn, rich.notebook_id)
    folder = settings.notebook_dir(rich.notebook_id)
    assert folder.is_dir()
    delete_notebook(conn, settings, rich.notebook_id)
    assert not folder.exists()
    for table in (
        "sources",
        "chunks",
        "notes",
        "conversations",
        "quizzes",
        "annotations",
        "memory_nodes",
        "audio_overviews",
    ):
        assert (
            _count(
                conn,
                f"SELECT count(*) FROM {table} WHERE notebook_id = ?",
                rich.notebook_id,
            )
            == 0
        ), table
    _gone(conn, settings, "Narwhalbody", "Okapiunique", "Borealis heliostat")


def test_secure_delete_is_on_for_every_connection_and_fts_table(conn):
    assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE sql LIKE 'CREATE VIRTUAL TABLE%fts5%'"
        )
    ]
    assert len(tables) >= 5
    for t in tables:
        assert (
            conn.execute(
                f"SELECT v FROM {t}_config WHERE k = 'secure-delete'"
            ).fetchone()[0]
            == 1
        ), t
