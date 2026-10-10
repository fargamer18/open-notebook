"""N+1 guard (Phase 16): page query counts must not grow with data size.

Counts only LiteNotebook's own SQL statements (SQLite reports FTS5's internal
shadow-table reads too; those start with ``--`` and are excluded).
"""

from __future__ import annotations

import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.notebooks import create_notebook
from litenotebook.notes import create_note
from litenotebook.web.app import create_app

PAGES = (
    "/notebooks/1/search?q=harbour+ledger",
    "/notebooks/1/notes",
    "/notebooks/1/memory",
    "/notebooks/1/study",
    "/notebooks/1",
)


def _counts(tmp_path, notes: int) -> dict[str, int]:  # type: ignore[no-untyped-def]
    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / f"d{notes}")})
    conn = open_db(settings.db_path)
    nb = create_notebook(conn, "Budget")
    pages = [f"Harbour ledger entry {i}. " * 20 for i in range(10)]
    ingest_pdf(
        conn,
        settings,
        nb,
        make_pdf(tmp_path / f"p{notes}.pdf", pages),
        FakeEmbedder(),
        FaissStore(conn, settings),
    )
    for i in range(notes):
        create_note(conn, nb.id, f"Note {i}", f"Harbour ledger note number {i}")
    conn.close()
    seen = {"n": 0}
    real = sqlite3.connect

    def counting(*a, **k):  # type: ignore[no-untyped-def]
        c = real(*a, **k)
        c.set_trace_callback(
            lambda sql: seen.__setitem__("n", seen["n"] + (not sql.startswith("--")))
        )
        return c

    app = create_app(
        settings,
        env={},
        embedder_factory=FakeEmbedder,
        llm_transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": []})
        ),
    )
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sqlite3, "connect", counting)
        with TestClient(app, base_url="http://127.0.0.1:5056") as client:
            for page in PAGES:
                client.get(page)  # warm caches
                seen["n"] = 0
                assert client.get(page).status_code == 200
                out[page] = seen["n"]
    return out


def test_page_queries_do_not_grow_with_the_number_of_notes(tmp_path):
    small = _counts(tmp_path, 5)
    large = _counts(tmp_path, 80)
    for page in PAGES:
        assert large[page] <= small[page] + 3, (page, small[page], large[page])
