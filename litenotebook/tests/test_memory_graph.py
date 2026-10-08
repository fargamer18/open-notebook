"""Phase 5: concept extraction, deterministic graph build, outcomes, migration."""

import shutil
import sqlite3

import pytest

from conftest import FakeEmbedder, make_pdf
from litenotebook.chat import (
    CitedChunk,
    complete_message,
    delete_conversation,
    end_message,
    start_exchange,
)
from litenotebook.db import available_migrations, connect, migrate
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source, ingest_pdf
from litenotebook.memory.concepts import extract_concepts, normalize_word, terms
from litenotebook.memory.graph import (
    RELATIONS,
    OutcomeError,
    backfill,
    check,
    get_outcome,
    rebuild,
    set_outcome,
    stats,
    sync_turn,
)
from litenotebook.notebooks import create_notebook, delete_notebook


def nodes(conn, notebook_id, node_type=None):
    sql = (
        "SELECT key, node_type, label, provenance FROM memory_nodes "
        "WHERE notebook_id = ?"
    )
    args = [notebook_id]
    if node_type:
        sql += " AND node_type = ?"
        args.append(node_type)
    return {r["key"]: dict(r) for r in conn.execute(sql + " ORDER BY id", args)}


def edges(conn, notebook_id):
    return {
        (r[0], r[1], r[2])
        for r in conn.execute(
            "SELECT a.key, e.relation, b.key FROM memory_edges e "
            "JOIN memory_nodes a ON a.id = e.from_node_id "
            "JOIN memory_nodes b ON b.id = e.to_node_id WHERE e.notebook_id = ?",
            (notebook_id,),
        )
    }


def snapshot(conn, notebook_id):
    return (
        sorted(nodes(conn, notebook_id).items()),
        sorted(edges(conn, notebook_id)),
        sorted(
            r[0]
            for r in conn.execute(
                "SELECT id FROM memory_nodes WHERE notebook_id = ?", (notebook_id,)
            )
        ),
    )


@pytest.fixture
def chunks(conn, settings, notebook, embedder, store, tmp_path):
    pdf = make_pdf(
        tmp_path / "telescope.pdf",
        [
            "The Borealis telescope has an aperture of 3.7 metres.",
            "Its calibration system Helios entered service in 2011.",
        ],
    )
    ingest_pdf(conn, settings, notebook, pdf, embedder, store)
    return conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks ORDER BY id"
    ).fetchall()


def turn(conn, notebook_id, question, answer, *, cid=None, cites=(), model="qwen"):
    ex = start_exchange(conn, notebook_id, cid, question)
    complete_message(
        conn,
        ex.assistant_message_id,
        answer,
        model=model,
        citations=[
            CitedChunk(
                i + 1,
                r["id"],
                r["source_id"],
                "telescope.pdf",
                r["page_no"],
                r["ord"],
                r["text"],
            )
            for i, r in enumerate(cites)
        ],
    )
    sync_turn(conn, ex.assistant_message_id)
    return ex


# --- concepts -------------------------------------------------------------------------


def test_concepts_drop_stopwords_and_keep_meaning():
    concepts = extract_concepts("What did we discuss about diagonalising matrix A?")
    assert concepts == ["diagonalising matrix", "diagonalising", "matrix"]
    for junk in ("what", "did", "we", "about", "discuss", "a"):
        assert junk not in concepts


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Eigenvalues, eigenvectors; EIGENVALUES!", ["eigenvalues", "eigenvectors"]),
        ("Zorbania’s capital", ["zorbania capital", "zorbania", "capital"]),
        ("ÉCOLE Polytechnique", ["école polytechnique", "école", "polytechnique"]),
        ("The aperture is 3.7 metres [S1] [M2].", ["aperture", "metres"]),
        ("2011 1642 12345", []),
        ("", []),
    ],
)
def test_concept_normalisation(text, expected):
    assert extract_concepts(text) == expected


def test_punctuation_breaks_phrases_and_limit_bounds_output():
    assert "telescope calibration" not in extract_concepts("telescope. Calibration")
    long = " ".join(f"concept{i} word{i}" for i in range(200))
    assert len(extract_concepts(long)) == 6
    assert normalize_word("Matrices’s") == "matrices"
    assert terms("what calibration system did we discuss?") == ["calibration", "system"]


# --- graph build ---------------------------------------------------------------------


def test_completed_turn_creates_expected_graph(conn, notebook, chunks):
    ex = turn(
        conn,
        notebook.id,
        "What is the Borealis telescope aperture?",
        "The aperture is 3.7 metres [S1].",
        cites=[chunks[0]],
    )
    aid, uid, cid = ex.assistant_message_id, ex.user_message_id, ex.conversation.id
    got = nodes(conn, notebook.id)
    assert got[f"question:{uid}"]["provenance"] == "user_statement"
    assert got[f"answer:{aid}"]["provenance"] == "source_backed"
    [ev] = [k for k in got if k.startswith("evidence:")]
    assert got[ev]["label"] == "telescope.pdf — page 1"
    assert {"concept:telescope aperture", "concept:aperture"} <= set(got)
    e = edges(conn, notebook.id)
    assert (f"conversation:{cid}", "contains", f"question:{uid}") in e
    assert (f"conversation:{cid}", "contains", f"answer:{aid}") in e
    assert (f"question:{uid}", "answered_by", f"answer:{aid}") in e
    assert (f"answer:{aid}", "supported_by", ev) in e
    assert (f"question:{uid}", "mentions", "concept:aperture") in e
    assert {r for _, r, _ in e} <= RELATIONS
    assert check(conn, notebook.id).ok


def test_follow_up_links_and_concepts_are_shared(conn, notebook):
    a = turn(conn, notebook.id, "What is the calibration system?", "It is Helios.")
    b = turn(
        conn,
        notebook.id,
        "When did the calibration system start?",
        "In 2011.",
        cid=a.conversation.id,
    )
    e = edges(conn, notebook.id)
    assert (
        f"question:{a.user_message_id}",
        "followed_by",
        f"question:{b.user_message_id}",
    ) in e
    calibration = [
        k for k in nodes(conn, notebook.id, "concept") if k == "concept:calibration"
    ]
    assert len(calibration) == 1  # reused, never duplicated
    assert len({(f, t) for f, r, t in e if t == "concept:calibration"}) == 2


def test_uncited_and_system_answers(conn, notebook):
    ex = turn(conn, notebook.id, "Anything?", "No sources.", model=None)
    got = nodes(conn, notebook.id)
    assert f"question:{ex.user_message_id}" in got
    assert f"answer:{ex.assistant_message_id}" not in got  # not an LLM answer
    ex2 = turn(conn, notebook.id, "Guess?", "Probably 42.")
    assert nodes(conn, notebook.id)[f"answer:{ex2.assistant_message_id}"][
        "provenance"
    ] == ("assistant_response")
    failed = start_exchange(conn, notebook.id, None, "Will this fail?")
    end_message(conn, failed.assistant_message_id, "failed", "LLM down")
    sync_turn(conn, failed.assistant_message_id)
    assert f"question:{failed.user_message_id}" in nodes(conn, notebook.id)
    assert f"answer:{failed.assistant_message_id}" not in nodes(conn, notebook.id)


def test_sync_and_rebuild_are_idempotent(conn, notebook, chunks):
    a = turn(conn, notebook.id, "Telescope aperture?", "3.7 m [S1].", cites=[chunks[0]])
    turn(conn, notebook.id, "Calibration system?", "Helios.", cid=a.conversation.id)
    set_outcome(conn, notebook.id, a.assistant_message_id, "useful")
    before = snapshot(conn, notebook.id)
    for _ in range(3):
        sync_turn(conn, a.assistant_message_id)
        assert backfill(conn) == 0
    assert snapshot(conn, notebook.id) == before
    rebuild(conn, notebook.id)
    rebuilt = snapshot(conn, notebook.id)
    assert rebuilt[0] == before[0] and rebuilt[1] == before[1]  # same content
    rebuild(conn, notebook.id)
    assert snapshot(conn, notebook.id)[:2] == before[:2]
    assert get_outcome(conn, a.assistant_message_id) == ("useful", None)  # kept


def test_source_deletion_keeps_historical_evidence(
    conn, settings, notebook, chunks, store
):
    ex = turn(conn, notebook.id, "Aperture?", "3.7 metres [S1].", cites=[chunks[0]])
    delete_source(conn, settings, notebook.id, chunks[0]["source_id"], store)
    sync_turn(conn, ex.assistant_message_id)
    rebuild(conn, notebook.id)
    got = nodes(conn, notebook.id, "source_evidence")
    [(key, ev)] = got.items()
    assert ev["provenance"] == "source_backed"
    content = conn.execute(
        "SELECT content FROM memory_nodes WHERE key = ?", (key,)
    ).fetchone()[0]
    assert content == "The Borealis telescope has an aperture of 3.7 metres."
    assert nodes(conn, notebook.id)[f"answer:{ex.assistant_message_id}"][
        "provenance"
    ] == ("source_backed")


def test_conversation_and_notebook_deletion_cascade(conn, settings, notebook):
    ex = turn(conn, notebook.id, "Calibration system?", "Helios.")
    other = turn(conn, notebook.id, "Telescope aperture?", "3.7 m.")
    delete_conversation(conn, notebook.id, ex.conversation.id)
    remaining = nodes(conn, notebook.id)
    assert not any(k.endswith(f":{ex.user_message_id}") for k in remaining)
    assert f"answer:{other.assistant_message_id}" in remaining
    assert check(conn, notebook.id).ok  # orphan concepts pruned on the next sync?
    sync_turn(conn, other.assistant_message_id)
    assert check(conn, notebook.id).ok
    delete_notebook(conn, settings, notebook.id)
    for table in ("memory_nodes", "memory_edges", "memory_outcomes"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_graph_is_notebook_scoped(conn, notebook):
    other = create_notebook(conn, "other")
    turn(conn, notebook.id, "Calibration system?", "Helios.")
    turn(conn, other.id, "Calibration system?", "Different answer.")
    mine, theirs = nodes(conn, notebook.id), nodes(conn, other.id)
    assert "concept:calibration" in mine and "concept:calibration" in theirs

    def ids(nb):
        sql = "SELECT id FROM memory_nodes WHERE notebook_id = ?"
        return {r[0] for r in conn.execute(sql, (nb,))}

    assert not ids(notebook.id) & ids(other.id)
    cross = conn.execute(
        "SELECT count(*) FROM memory_edges e "
        "JOIN memory_nodes a ON a.id = e.from_node_id "
        "JOIN memory_nodes b ON b.id = e.to_node_id "
        "WHERE a.notebook_id <> b.notebook_id OR a.notebook_id <> e.notebook_id"
    ).fetchone()[0]
    assert cross == 0


def test_closed_relation_vocabulary_enforced_by_schema(conn, notebook):
    ex = turn(conn, notebook.id, "Calibration system?", "Helios.")
    ids = [r[0] for r in conn.execute("SELECT id FROM memory_nodes LIMIT 2")]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO memory_edges (notebook_id, from_node_id, to_node_id, "
            "relation, "
            "provenance) VALUES (?, ?, ?, 'model_says_so', 'x')",
            (notebook.id, ids[0], ids[1]),
        )
    with pytest.raises(sqlite3.IntegrityError):  # duplicate edge
        row = conn.execute("SELECT * FROM memory_edges LIMIT 1").fetchone()
        conn.execute(
            "INSERT INTO memory_edges (notebook_id, from_node_id, to_node_id, "
            "relation, "
            "provenance) VALUES (?, ?, ?, ?, 'again')",
            (
                row["notebook_id"],
                row["from_node_id"],
                row["to_node_id"],
                row["relation"],
            ),
        )
    with pytest.raises(sqlite3.IntegrityError):  # duplicate node key
        conn.execute(
            "INSERT INTO memory_nodes (notebook_id, key, node_type, label, provenance) "
            "VALUES (?, 'concept:helios', 'concept', 'x', 'derived')",
            (notebook.id,),
        )
    assert ex


# --- outcomes ----------------------------------------------------------------------


def test_outcomes_and_corrections(conn, notebook):
    ex = turn(
        conn,
        notebook.id,
        "When did Helios enter service?",
        "Helios entered service in 2010.",
    )
    aid = ex.assistant_message_id
    set_outcome(conn, notebook.id, aid, "useful")
    assert get_outcome(conn, aid) == ("useful", None)
    with pytest.raises(OutcomeError, match="Enter the correction"):
        set_outcome(conn, notebook.id, aid, "corrected", "   ")
    set_outcome(conn, notebook.id, aid, "corrected", "Helios entered service in 2011.")
    assert get_outcome(conn, aid) == ("corrected", "Helios entered service in 2011.")
    original = conn.execute(
        "SELECT content FROM messages WHERE id = ?", (aid,)
    ).fetchone()
    assert original[0] == "Helios entered service in 2010."  # history preserved
    got = nodes(conn, notebook.id)
    assert got[f"correction:{aid}"]["provenance"] == "user_correction"
    assert (f"correction:{aid}", "corrects", f"answer:{aid}") in edges(
        conn, notebook.id
    )
    set_outcome(conn, notebook.id, aid, "dead_end")
    assert f"correction:{aid}" not in nodes(conn, notebook.id)
    set_outcome(conn, notebook.id, aid, None)
    assert get_outcome(conn, aid) is None
    assert stats(conn, notebook.id).outcomes == {}


def test_outcome_validation_and_scope(conn, notebook):
    other = create_notebook(conn, "other")
    ex = turn(conn, notebook.id, "Q?", "A.")
    from litenotebook.chat import MessageNotFoundError

    with pytest.raises(MessageNotFoundError):
        set_outcome(conn, other.id, ex.assistant_message_id, "useful")
    with pytest.raises(OutcomeError):
        set_outcome(conn, notebook.id, ex.assistant_message_id, "brilliant")
    with pytest.raises(OutcomeError):
        set_outcome(conn, notebook.id, ex.assistant_message_id, "corrected", "x" * 2001)
    system = turn(conn, notebook.id, "No sources?", "None.", model=None)
    with pytest.raises(OutcomeError, match="Only completed model answers"):
        set_outcome(conn, notebook.id, system.assistant_message_id, "useful")
    with pytest.raises(MessageNotFoundError):  # a user message is not an answer
        set_outcome(conn, notebook.id, ex.user_message_id, "useful")


def test_corrections_survive_rebuild(conn, notebook):
    ex = turn(conn, notebook.id, "When did Helios enter service?", "In 2010.")
    set_outcome(conn, notebook.id, ex.assistant_message_id, "corrected", "In 2011.")
    rebuild(conn, notebook.id)
    assert f"correction:{ex.assistant_message_id}" in nodes(conn, notebook.id)
    assert check(conn, notebook.id).ok


# --- migration & backfill -----------------------------------------------------------


def test_phase5_migration_and_backfill_of_populated_phase4_db(
    settings, tmp_path, monkeypatch
):
    from litenotebook import db

    partial = tmp_path / "migrations_4"
    partial.mkdir()
    for _v, path in available_migrations()[:4]:
        shutil.copy(path, partial / path.name)
    conn = connect(settings.db_path)
    with monkeypatch.context() as m:
        m.setattr(db, "MIGRATIONS_DIR", partial)
        assert migrate(conn) == 4
    nb = create_notebook(conn, "Maths")
    pdf = make_pdf(tmp_path / "m.pdf", ["Diagonalising a matrix needs eigenvectors."])
    store = FaissStore(conn, settings)
    ingest_pdf(conn, settings, nb, pdf, FakeEmbedder(), store)
    chunk = conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks"
    ).fetchone()
    # Phase 4-era rows, written the way Phase 4 code wrote them
    ex = start_exchange(conn, nb.id, None, "How do we diagonalise matrix A?")
    conn.execute(
        "UPDATE messages SET status = 'complete', content = ?, model = 'qwen' "
        "WHERE id = ?",
        ("Use eigenvectors [S1].", ex.assistant_message_id),
    )
    conn.execute(
        "INSERT INTO message_citations (message_id, label, ordinal, chunk_id, "
        "source_id, source_filename, page_no, chunk_ord, chunk_text) "
        "VALUES (?, 1, 0, ?, ?, 'm.pdf', ?, ?, ?)",
        (ex.assistant_message_id, chunk[0], chunk[1], chunk[2], chunk[3], chunk[4]),
    )
    conn.commit()
    before = {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in (
            "notebooks",
            "sources",
            "chunks",
            "conversations",
            "messages",
            "message_citations",
        )
    }

    assert migrate(conn) == len(available_migrations())
    after = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in before}
    assert after == before
    assert conn.execute("SELECT count(*) FROM memory_nodes").fetchone()[0] == 0
    assert backfill(conn) == 1  # what the app does at startup; no LLM involved
    assert backfill(conn) == 0  # idempotent
    got = nodes(conn, nb.id)
    assert (
        f"answer:{ex.assistant_message_id}" in got
        and "concept:diagonalise matrix" in got
    )
    assert check(conn, nb.id).ok and store.check(nb.id).ok
    conn.close()
