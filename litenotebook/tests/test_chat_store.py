"""Chat persistence: conversations, messages, citation snapshots, lifecycle."""

import shutil

import pytest

from conftest import make_pdf
from litenotebook.chat import (
    INTERRUPTED_MESSAGE,
    RESTART_MESSAGE,
    ChatBusyError,
    ChatError,
    CitationNotFoundError,
    CitedChunk,
    ConversationNotFoundError,
    MessageNotFoundError,
    clean_question,
    complete_message,
    create_conversation,
    delete_conversation,
    end_message,
    get_chunk_evidence,
    get_citation,
    get_conversation,
    get_message,
    history_turns,
    list_conversations,
    list_messages,
    reconcile_generating,
    recover_generating,
    rename_conversation,
    start_exchange,
    start_retry,
    title_from,
)
from litenotebook.db import available_migrations, connect, migrate
from litenotebook.ingest.pipeline import delete_source, ingest_pdf
from litenotebook.notebooks import create_notebook, delete_notebook
from litenotebook.services.settings import SettingsService


def chunk_rows(conn, notebook_id):
    return conn.execute(
        "SELECT c.id, c.source_id, c.page_no, c.ord, c.text, s.original_filename "
        "FROM chunks c JOIN sources s ON s.id = c.source_id "
        "WHERE c.notebook_id = ? ORDER BY c.id",
        (notebook_id,),
    ).fetchall()


def cited(row, label):
    return CitedChunk(
        label,
        row["id"],
        row["source_id"],
        row["original_filename"],
        row["page_no"],
        row["ord"],
        row["text"],
    )


@pytest.fixture
def ingested(conn, settings, notebook, embedder, store, tmp_path):
    pdf = make_pdf(
        tmp_path / "linear.pdf",
        ["Rank equals the number of pivots.", "Quillport is the capital of Zorbania."],
    )
    ingest_pdf(conn, settings, notebook, pdf, embedder, store)
    return chunk_rows(conn, notebook.id)


# --- titles & validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "title"),
    [
        ("What is rank?", "What is rank?"),
        ("  multiple   spaces\n\nand lines  ", "multiple spaces and lines"),
        (
            "Explain in detail how Gaussian elimination produces the row echelon "
            "form of a matrix",
            "Explain in detail how Gaussian elimination produces the row…",
        ),
        ("x" * 100, "x" * 60 + "…"),
    ],
)
def test_deterministic_titles(question, title):
    assert title_from(question) == title
    assert title_from(question) == title_from(question)


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "x" * 4001, "nul\x00byte"])
def test_invalid_questions(bad):
    with pytest.raises(ChatError):
        clean_question(bad)


# --- conversations -------------------------------------------------------------------


def test_conversation_crud_and_isolation(conn, notebook):
    other = create_notebook(conn, "other")
    a = create_conversation(conn, notebook.id, "First")
    b = create_conversation(conn, notebook.id, "Second")
    create_conversation(conn, other.id, "Elsewhere")
    assert {c.title for c in list_conversations(conn, notebook.id)} == {
        "First",
        "Second",
    }
    assert (
        rename_conversation(conn, notebook.id, a.id, "  Renamed  ").title == "Renamed"
    )
    with pytest.raises(ChatError):
        rename_conversation(conn, notebook.id, a.id, "   ")
    with pytest.raises(ConversationNotFoundError):
        get_conversation(conn, other.id, a.id)
    with pytest.raises(ConversationNotFoundError):
        rename_conversation(conn, other.id, a.id, "hijack")
    with pytest.raises(ConversationNotFoundError):
        delete_conversation(conn, other.id, b.id)
    delete_conversation(conn, notebook.id, b.id)
    assert [c.title for c in list_conversations(conn, notebook.id)] == ["Renamed"]


def test_exchange_creates_conversation_lazily(conn, notebook):
    ex = start_exchange(conn, notebook.id, None, "What is the rank of A?")
    assert ex.created_conversation and ex.conversation.title == "What is the rank of A?"
    messages = list_messages(conn, ex.conversation.id)
    assert [(m.role, m.status) for m in messages] == [
        ("user", "complete"),
        ("assistant", "generating"),
    ]
    assert messages[1].reply_to == messages[0].id == ex.user_message_id
    again = start_exchange(conn, notebook.id, ex.conversation.id, "Follow-up?")
    assert not again.created_conversation
    assert len(list_conversations(conn, notebook.id)) == 1
    other = create_notebook(conn, "other")
    with pytest.raises(ConversationNotFoundError):
        start_exchange(conn, other.id, ex.conversation.id, "sneaky")


def test_complete_message_persists_citation_snapshots(conn, notebook, ingested):
    ex = start_exchange(conn, notebook.id, None, "Capital?")
    assert complete_message(
        conn,
        ex.assistant_message_id,
        "Quillport [S2] (see [S2]).",
        model="qwen",
        citations=[cited(ingested[1], 2)],
        meta={"timings": {"x": 1}},
    )
    message = get_message(conn, notebook.id, ex.assistant_message_id)
    assert message.status == "complete" and message.model == "qwen"
    [c] = message.citations
    assert (c.label, c.chunk_id, c.page_no, c.source_filename) == (
        2,
        ingested[1]["id"],
        2,
        "linear.pdf",
    )
    assert c.chunk_text == ingested[1]["text"] and c.chunk_exists and c.source_exists
    assert get_citation(conn, notebook.id, c.id) == c
    # only generating messages can be completed
    assert not complete_message(conn, ex.assistant_message_id, "again", model="qwen")


def test_deleted_source_keeps_citation_snapshot(
    conn, settings, notebook, ingested, store
):
    ex = start_exchange(conn, notebook.id, None, "Capital?")
    complete_message(
        conn,
        ex.assistant_message_id,
        "Quillport [S1].",
        model="qwen",
        citations=[cited(ingested[1], 1)],
    )
    delete_source(conn, settings, notebook.id, ingested[1]["source_id"], store)
    [c] = get_message(conn, notebook.id, ex.assistant_message_id).citations
    assert not c.chunk_exists and not c.source_exists
    assert c.chunk_text == "Quillport is the capital of Zorbania."
    assert c.source_filename == "linear.pdf" and c.page_no == 2
    assert get_chunk_evidence(conn, notebook.id, c.chunk_id) is None


def test_citation_and_message_access_is_notebook_scoped(conn, notebook, ingested):
    other = create_notebook(conn, "other")
    ex = start_exchange(conn, notebook.id, None, "Capital?")
    complete_message(
        conn,
        ex.assistant_message_id,
        "x [S1]",
        model="q",
        citations=[cited(ingested[0], 1)],
    )
    cid = get_message(conn, notebook.id, ex.assistant_message_id).citations[0].id
    with pytest.raises(MessageNotFoundError):
        get_message(conn, other.id, ex.assistant_message_id)
    with pytest.raises(CitationNotFoundError):
        get_citation(conn, other.id, cid)
    assert get_chunk_evidence(conn, other.id, ingested[0]["id"]) is None


def test_chunk_evidence_neighbours_stay_in_source(
    conn, settings, notebook, ingested, embedder, store, tmp_path
):
    second = make_pdf(tmp_path / "b.pdf", ["Another document entirely."])
    ingest_pdf(conn, settings, notebook, second, embedder, store)
    first, last = ingested[0], ingested[-1]
    ev = get_chunk_evidence(conn, notebook.id, first["id"])
    assert ev.prev_id is None and ev.next_id == ingested[1]["id"]
    ev_last = get_chunk_evidence(conn, notebook.id, last["id"])
    assert ev_last.next_id is None  # does not cross into b.pdf
    conn.execute(
        "UPDATE sources SET status = 'failed' WHERE id = ?", (first["source_id"],)
    )
    conn.commit()
    assert get_chunk_evidence(conn, notebook.id, first["id"]) is None  # not ready


def test_failure_retry_lifecycle_never_duplicates(conn, notebook):
    ex = start_exchange(conn, notebook.id, None, "Q?")
    assert end_message(conn, ex.assistant_message_id, "failed", "LLM down", content="")
    with pytest.raises(ChatBusyError):
        start_retry(conn, notebook.id, ex.user_message_id)  # not an assistant answer
    retry = start_retry(conn, notebook.id, ex.assistant_message_id)
    assert retry.assistant_message_id == ex.assistant_message_id
    assert retry.user_message_id == ex.user_message_id and retry.question == "Q?"
    messages = list_messages(conn, ex.conversation.id)
    assert len(messages) == 2 and messages[1].status == "generating"
    complete_message(conn, ex.assistant_message_id, "Done.", model="q")
    with pytest.raises(ChatBusyError):
        start_retry(conn, notebook.id, ex.assistant_message_id)  # complete
    other = create_notebook(conn, "other")
    with pytest.raises(MessageNotFoundError):
        start_retry(conn, other.id, ex.assistant_message_id)


def test_restart_recovery_and_reconcile(conn, notebook):
    a = start_exchange(conn, notebook.id, None, "One?")
    b = start_exchange(conn, notebook.id, a.conversation.id, "Two?")
    assert reconcile_generating(conn, {b.assistant_message_id}) == 1
    first = get_message(conn, notebook.id, a.assistant_message_id)
    assert first.status == "interrupted" and first.error == INTERRUPTED_MESSAGE
    assert recover_generating(conn) == 1
    second = get_message(conn, notebook.id, b.assistant_message_id)
    assert second.status == "interrupted" and second.error == RESTART_MESSAGE
    assert recover_generating(conn) == 0


def test_history_only_includes_completed_llm_turns(conn, notebook):
    a = start_exchange(conn, notebook.id, None, "First?")
    complete_message(conn, a.assistant_message_id, "One [S1].", model="q")
    b = start_exchange(conn, notebook.id, a.conversation.id, "Second?")
    end_message(conn, b.assistant_message_id, "failed", "boom")
    c = start_exchange(conn, notebook.id, a.conversation.id, "Third?")
    complete_message(conn, c.assistant_message_id, "No sources.", model=None)
    d = start_exchange(conn, notebook.id, a.conversation.id, "Fourth?")
    turns = history_turns(conn, a.conversation.id, d.user_message_id)
    assert [(t.question, t.answer) for t in turns] == [("First?", "One [S1].")]
    assert history_turns(conn, a.conversation.id, a.user_message_id) == []


def test_delete_conversation_cascades_and_refuses_while_generating(
    conn, notebook, ingested
):
    ex = start_exchange(conn, notebook.id, None, "Q?")
    with pytest.raises(ChatBusyError):
        delete_conversation(conn, notebook.id, ex.conversation.id)
    complete_message(
        conn,
        ex.assistant_message_id,
        "A [S1]",
        model="q",
        citations=[cited(ingested[0], 1)],
    )
    delete_conversation(conn, notebook.id, ex.conversation.id)
    for table in ("messages", "message_citations", "conversations"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == len(ingested)


def test_notebook_delete_cascades_chat(conn, settings, notebook, ingested):
    ex = start_exchange(conn, notebook.id, None, "Q?")
    complete_message(
        conn,
        ex.assistant_message_id,
        "A [S1]",
        model="q",
        citations=[cited(ingested[0], 1)],
    )
    delete_notebook(conn, settings, notebook.id)
    for table in ("messages", "message_citations", "conversations"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_one_reply_per_question_enforced_by_schema(conn, notebook):
    import sqlite3

    ex = start_exchange(conn, notebook.id, None, "Q?")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO messages (conversation_id, role, status, reply_to) "
            "VALUES (?, 'assistant', 'generating', ?)",
            (ex.conversation.id, ex.user_message_id),
        )


# --- migration ----------------------------------------------------------------


def test_phase4_migration_preserves_populated_phase3_db(
    settings, embedder, tmp_path, monkeypatch
):
    from litenotebook import db
    from litenotebook.index.faiss_store import FaissStore

    partial = tmp_path / "migrations_3"
    partial.mkdir()
    for _v, path in available_migrations()[:3]:
        shutil.copy(path, partial / path.name)
    conn = connect(settings.db_path)
    with monkeypatch.context() as m:
        m.setattr(db, "MIGRATIONS_DIR", partial)
        assert migrate(conn) == 3
    nb = create_notebook(conn, "Maths")
    pdf = make_pdf(tmp_path / "m.pdf", ["Pivots and rank.", "Eigenvalues."])
    store = FaissStore(conn, settings)
    ingest_pdf(conn, settings, nb, pdf, embedder, store)
    SettingsService(conn, env={}).update(rag_top_k=5)
    before = {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("notebooks", "sources", "pages", "chunks", "settings")
    }

    assert migrate(conn) == len(available_migrations()) >= 4
    after = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in before}
    assert after == before
    assert conn.execute("SELECT name FROM notebooks").fetchone()[0] == "Maths"
    assert conn.execute("SELECT status FROM sources").fetchone()[0] == "ready"
    assert SettingsService(conn, env={}).effective().rag_top_k == 5
    assert store.check(nb.id).ok
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    ex = start_exchange(conn, nb.id, None, "Works after upgrade?")
    assert ex.conversation.notebook_id == nb.id
    conn.close()
