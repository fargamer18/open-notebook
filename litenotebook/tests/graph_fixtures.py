"""A realistic notebook for knowledge-graph tests (no LLM, no MiniLM).

Two PDFs, two conversations with cited answers (one answer corrected, one
marked useful), two notes (one linked to a passage, one to a Reader
annotation), and a quiz with one attempt in which the "quillport" question was
missed — so that concept carries real "Needs review" study data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from conftest import FakeEmbedder, make_pdf
from litenotebook.chat import CitedChunk, complete_message, start_exchange
from litenotebook.config import Settings
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.memory.graph import set_outcome, sync_turn
from litenotebook.notebooks import create_notebook
from litenotebook.notes import create_note, link_annotation, link_chunk
from litenotebook.reader.annotations import create_annotation
from litenotebook.study import store
from quiz_fixtures import PDF_A, PDF_B, correct_choice, make_ready_quiz, wrong_choice


def _chunk(conn: sqlite3.Connection, source_id: int, page: int = 1) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks WHERE source_id = ? "
        "AND page_no = ? ORDER BY ord LIMIT 1",
        (source_id, page),
    ).fetchone()
    assert row is not None
    return row


def _exchange(
    conn: sqlite3.Connection,
    notebook_id: int,
    conversation_id: int | None,
    question: str,
    answer: str,
    chunk: sqlite3.Row,
    filename: str,
) -> tuple[int, int]:
    ex = start_exchange(conn, notebook_id, conversation_id, question)
    complete_message(
        conn,
        ex.assistant_message_id,
        answer,
        model="fixture",
        citations=[
            CitedChunk(
                1,
                chunk["id"],
                chunk["source_id"],
                filename,
                chunk["page_no"],
                chunk["ord"],
                chunk["text"],
            )
        ],
    )
    sync_turn(conn, ex.assistant_message_id)
    return ex.conversation.id, ex.assistant_message_id


def build_graph_notebook(
    conn: sqlite3.Connection, settings: Settings, tmp: Path, name: str = "Graph study"
) -> dict[str, Any]:
    nb = create_notebook(conn, name)
    store_ = FaissStore(conn, settings)
    atlas = ingest_pdf(
        conn, settings, nb, make_pdf(tmp / "Atlas.pdf", PDF_A), FakeEmbedder(), store_
    ).source_id
    algebra = ingest_pdf(
        conn, settings, nb, make_pdf(tmp / "Algebra.pdf", PDF_B), FakeEmbedder(), store_
    ).source_id

    geo, capital_answer = _exchange(
        conn,
        nb.id,
        None,
        "What is the capital of Zorbania?",
        "Quillport is the harbour capital of Zorbania [S1].",
        _chunk(conn, atlas, 1),
        "Atlas.pdf",
    )
    _, river_answer = _exchange(
        conn,
        nb.id,
        geo,
        "Where does the Velt river rise?",
        "The Velt river rises in the Grey Sea [S1].",
        _chunk(conn, atlas, 2),
        "Atlas.pdf",
    )
    set_outcome(
        conn,
        nb.id,
        river_answer,
        "corrected",
        "The Velt river rises in the Orrin mountains and flows into the Grey Sea.",
    )
    algebra_conv, matrix_answer = _exchange(
        conn,
        nb.id,
        None,
        "When is a square matrix diagonalisable?",
        "A square matrix is diagonalisable with a full set of eigenvectors [S1].",
        _chunk(conn, algebra, 1),
        "Algebra.pdf",
    )
    set_outcome(conn, nb.id, matrix_answer, "useful")

    trip = create_note(
        conn, nb.id, "Zorbania trip", "Quillport harbour, blue glass and the Velt."
    )
    link_chunk(conn, nb.id, trip.id, _chunk(conn, atlas, 1)["id"])
    ann = create_annotation(
        conn,
        settings,
        nb.id,
        algebra,
        page_no=1,
        selected_text="full set of linearly independent eigenvectors",
        comment="key condition",
        rects=[[0.1, 0.1, 0.5, 0.12]],
    )
    revision = create_note(
        conn, nb.id, "Matrix revision", "Diagonalisation needs eigenvectors."
    )
    link_annotation(conn, nb.id, revision.id, ann.id)

    quiz = make_ready_quiz(conn, nb.id, atlas, title="Zorbania quiz")
    attempt = store.start_attempt(conn, nb.id, quiz)
    store.answer(conn, nb.id, attempt, 1, correct_choice(conn, attempt, 1))
    store.answer(conn, nb.id, attempt, 2, correct_choice(conn, attempt, 2))
    store.answer(conn, nb.id, attempt, 3, wrong_choice(conn, attempt, 3))

    def node(key: str) -> int:
        row = conn.execute(
            "SELECT id FROM memory_nodes WHERE notebook_id = ? AND key = ?",
            (nb.id, key),
        ).fetchone()
        assert row is not None, key
        return int(row[0])

    return {
        "nb": nb.id,
        "atlas": atlas,
        "algebra": algebra,
        "geo_conv": geo,
        "algebra_conv": algebra_conv,
        "capital_answer": capital_answer,
        "river_answer": river_answer,
        "matrix_answer": matrix_answer,
        "trip_note": trip.id,
        "revision_note": revision.id,
        "annotation": ann.id,
        "quiz": quiz,
        "attempt": attempt,
        "quillport": node("concept:quillport"),
        "geo_node": node(f"conversation:{geo}"),
        "trip_node": node(f"note:{trip.id}"),
        "correction_node": node(f"correction:{river_answer}"),
    }
