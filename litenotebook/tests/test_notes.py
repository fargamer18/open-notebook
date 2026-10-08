"""Phase 7: notes — service, links/snapshots, graph memory, web, security."""

import re

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.chat import CitedChunk, complete_message, start_exchange
from litenotebook.db import connect, open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source, ingest_pdf
from litenotebook.memory.graph import check, rebuild, set_outcome, sync_turn
from litenotebook.memory.search import search_memory
from litenotebook.notebooks import create_notebook, delete_notebook
from litenotebook.notes import (
    MAX_CONTENT_BYTES,
    NoteConflictError,
    NoteError,
    NoteNotFoundError,
    create_note,
    delete_note,
    get_note,
    link_chunk,
    link_citation,
    link_memory_node,
    link_message,
    list_links,
    list_notes,
    note_concepts,
    remove_link,
    update_note,
)
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
ORIGIN = {"Origin": BASE}
HX = {**ORIGIN, "HX-Request": "true"}


def keys(conn, notebook_id):
    return {
        r[0]
        for r in conn.execute(
            "SELECT key FROM memory_nodes WHERE notebook_id = ?", (notebook_id,)
        )
    }


@pytest.fixture
def world(conn, settings, notebook, embedder, store, tmp_path):
    pdf = make_pdf(
        tmp_path / "eig.pdf",
        [
            "Diagonalising a matrix needs a full set of eigenvectors.",
            "The rank equals the number of pivots.",
        ],
    )
    ingest_pdf(conn, settings, notebook, pdf, embedder, store)
    chunk = conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks ORDER BY id"
    ).fetchone()
    ex = start_exchange(conn, notebook.id, None, "How do we diagonalise a matrix?")
    complete_message(
        conn,
        ex.assistant_message_id,
        "Find eigenvectors [S1].",
        model="qwen",
        citations=[
            CitedChunk(1, chunk[0], chunk[1], "eig.pdf", chunk[2], chunk[3], chunk[4])
        ],
    )
    sync_turn(conn, ex.assistant_message_id)
    return {"chunk": chunk, "ex": ex}


# --- service ---------------------------------------------------------------------


def test_create_edit_rename_and_timestamps(conn, notebook):
    note = create_note(conn, notebook.id, "  Linear   Algebra Review ", "First line.")
    assert note.title == "Linear Algebra Review" and note.revision == 1
    assert note.created_at == note.updated_at
    edited = update_note(
        conn, notebook.id, note.id, content="First line.\nSecond.", base_revision=1
    )
    assert edited.revision == 2 and edited.content.endswith("Second.")
    renamed = update_note(
        conn, notebook.id, note.id, title="LA review", base_revision=2
    )
    assert renamed.title == "LA review" and renamed.updated_at >= note.updated_at
    same = update_note(conn, notebook.id, note.id, title="LA review", base_revision=3)
    assert same.revision == 3  # no-op save does not bump
    with pytest.raises(NoteConflictError):
        update_note(conn, notebook.id, note.id, content="stale", base_revision=1)
    assert get_note(conn, notebook.id, note.id).content.endswith("Second.")
    assert [n.title for n in list_notes(conn, notebook.id)] == ["LA review"]


@pytest.mark.parametrize("title", ["", "   ", "x" * 201, "\x00\x01"])
def test_invalid_titles(conn, notebook, title):
    with pytest.raises(NoteError):
        create_note(conn, notebook.id, title)


def test_content_size_limit(conn, notebook):
    note = create_note(conn, notebook.id, "Big", "a" * MAX_CONTENT_BYTES)
    with pytest.raises(NoteError, match="1 MiB"):
        update_note(
            conn, notebook.id, note.id, content="é" * (MAX_CONTENT_BYTES // 2 + 1)
        )
    with pytest.raises(NoteError, match="1 MiB"):
        create_note(conn, notebook.id, "Too big", "b" * (MAX_CONTENT_BYTES + 1))


def test_notes_are_notebook_scoped(conn, notebook):
    other = create_notebook(conn, "other")
    note = create_note(conn, notebook.id, "Mine")
    for fn in (
        lambda: get_note(conn, other.id, note.id),
        lambda: update_note(conn, other.id, note.id, content="x"),
        lambda: delete_note(conn, other.id, note.id),
    ):
        with pytest.raises(NoteNotFoundError):
            fn()
    assert list_notes(conn, other.id) == []


def test_links_snapshot_and_survive_deletion(conn, settings, notebook, world, store):
    note = create_note(conn, notebook.id, "Eigen notes")
    chunk, ex = world["chunk"], world["ex"]
    link_chunk(conn, notebook.id, note.id, chunk[0])
    link_chunk(conn, notebook.id, note.id, chunk[0])  # idempotent
    link_message(conn, notebook.id, note.id, ex.assistant_message_id)
    concept = conn.execute(
        "SELECT id FROM memory_nodes WHERE key = 'concept:eigenvectors'"
    ).fetchone()[0]
    link_memory_node(conn, notebook.id, note.id, concept)
    links = list_links(conn, notebook.id, note.id)
    assert [x.link_type for x in links] == ["source_chunk", "message", "memory_node"]
    assert all(x.exists for x in links)
    assert links[0].label == "eig.pdf" and links[0].page_no == 1
    assert links[1].citations[0]["filename"] == "eig.pdf"
    delete_source(conn, settings, notebook.id, chunk[1], store)
    from litenotebook.chat import delete_conversation

    delete_conversation(conn, notebook.id, ex.conversation.id)
    after = list_links(conn, notebook.id, note.id)
    assert [x.exists for x in after[:2]] == [False, False]
    assert after[0].excerpt == chunk[4] and "eigenvectors" in after[1].excerpt
    assert get_note(conn, notebook.id, note.id).title == "Eigen notes"


def test_citation_link_works_after_source_deletion(
    conn, settings, notebook, world, store
):
    from litenotebook.chat import get_message

    [cit] = get_message(conn, notebook.id, world["ex"].assistant_message_id).citations
    delete_source(conn, settings, notebook.id, cit.source_id, store)
    note = create_note(conn, notebook.id, "From a deleted source")
    link_citation(conn, notebook.id, note.id, cit.id)
    [link] = list_links(conn, notebook.id, note.id)
    assert not link.exists and link.excerpt == cit.chunk_text


def test_links_are_scoped(conn, notebook, world):
    other = create_notebook(conn, "other")
    theirs = create_note(conn, other.id, "Theirs")
    with pytest.raises(NoteNotFoundError):
        link_chunk(conn, other.id, theirs.id, world["chunk"][0])
    from litenotebook.chat import MessageNotFoundError

    with pytest.raises(MessageNotFoundError):
        link_message(conn, other.id, theirs.id, world["ex"].assistant_message_id)


# --- graph memory ----------------------------------------------------------------


def test_note_concepts_graph_and_cleanup(conn, notebook, world):
    note = create_note(
        conn,
        notebook.id,
        "Diagonalisation review",
        "Eigenvectors make diagonalisation possible.",
    )
    assert "eigenvectors" in note_concepts(conn, notebook.id, note.id)
    assert f"note:{note.id}" in keys(conn, notebook.id)
    link_chunk(conn, notebook.id, note.id, world["chunk"][0])
    assert any(k.startswith("note_evidence:") for k in keys(conn, notebook.id))
    unique = create_note(conn, notebook.id, "Zebra crossing", "Quaternions zebra.")
    assert "concept:zebra crossing" in keys(conn, notebook.id)
    update_note(conn, notebook.id, unique.id, title="Plain title", content="nothing")
    assert "concept:zebra crossing" not in keys(conn, notebook.id)  # re-derived
    delete_note(conn, notebook.id, note.id)
    remaining = keys(conn, notebook.id)
    assert f"note:{note.id}" not in remaining
    assert not any(k.startswith("note_evidence:") for k in remaining)
    assert "concept:eigenvectors" in remaining  # still used by the chat answer
    assert check(conn, notebook.id).ok


def test_rebuild_includes_notes_and_is_idempotent(conn, notebook, world):
    note = create_note(conn, notebook.id, "Rank review", "Rank equals pivots.")
    link_chunk(conn, notebook.id, note.id, world["chunk"][0])
    set_outcome(conn, notebook.id, world["ex"].assistant_message_id, "useful")
    before = sorted(keys(conn, notebook.id))
    for _ in range(2):
        rebuild(conn, notebook.id)
        assert sorted(keys(conn, notebook.id)) == before
    report = check(conn, notebook.id)
    assert report.ok and report.notes == report.synced_notes == 1
    assert len(list_links(conn, notebook.id, note.id)) == 1  # never touched


def test_notes_rank_as_user_notes_in_memory(conn, notebook, world):
    create_note(conn, notebook.id, "Pivot facts", "The rank equals the pivots count.")
    items = search_memory(conn, notebook.id, "rank pivots").items
    note = next(i for i in items if i.kind == "note")
    assert note.trust == "user note" and note.provenance == "user_note"
    assert note.question == "Pivot facts"


def test_note_relations_come_from_closed_vocabulary(conn, notebook, world):
    note = create_note(conn, notebook.id, "Eigen", "Eigenvectors.")
    link_message(conn, notebook.id, note.id, world["ex"].assistant_message_id)
    rels = {
        r[0]
        for r in conn.execute(
            "SELECT relation FROM memory_edges WHERE provenance = ?",
            (f"note:{note.id}",),
        )
    }
    assert rels == {"note_mentions", "note_supported_by"}


def test_remove_link_and_notebook_delete_cascade(conn, settings, notebook, world):
    note = create_note(conn, notebook.id, "Temp")
    link_id = link_chunk(conn, notebook.id, note.id, world["chunk"][0])
    remove_link(conn, notebook.id, note.id, link_id)
    assert list_links(conn, notebook.id, note.id) == []
    delete_notebook(conn, settings, notebook.id)
    for table in ("notes", "note_links", "memory_nodes", "notes_fts"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


# --- web --------------------------------------------------------------------------


@pytest.fixture
def web(tmp_path):
    from litenotebook.config import load_settings

    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    conn = open_db(s.db_path)
    nb = create_notebook(conn, "Research")
    ingest_pdf(
        conn,
        s,
        nb,
        make_pdf(tmp_path / "e.pdf", ["Eigenvectors and pivots."]),
        FakeEmbedder(),
        FaissStore(conn, s),
    )
    ex = start_exchange(conn, nb.id, None, "What are eigenvectors?")
    chunk = conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks"
    ).fetchone()
    complete_message(
        conn,
        ex.assistant_message_id,
        "Special vectors [S1] [M9].",
        model="qwen",
        citations=[
            CitedChunk(1, chunk[0], chunk[1], "e.pdf", chunk[2], chunk[3], chunk[4])
        ],
    )
    sync_turn(conn, ex.assistant_message_id)
    conn.close()
    calls = []

    def factory():
        calls.append(1)
        return FakeEmbedder()

    llama_calls = []

    def llama(r):
        llama_calls.append(r)
        return httpx.Response(200, json={"data": []})

    app = create_app(
        s, env={}, llm_transport=httpx.MockTransport(llama), embedder_factory=factory
    )
    with TestClient(app, base_url=BASE) as c:
        yield {
            "c": c,
            "s": s,
            "nb": nb.id,
            "ex": ex,
            "chunk": chunk,
            "embedder_calls": calls,
            "llama": llama_calls,
        }


def test_notes_ui_flow(web):
    c, nb = web["c"], web["nb"]
    assert "No notes yet." in c.get(f"/notebooks/{nb}/notes").text
    r = c.post(
        f"/notebooks/{nb}/notes",
        data={"title": "Linear Algebra Review"},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert r.status_code == 303 and r.headers["location"] == f"/notebooks/{nb}/notes/1"
    page = c.get(f"/notebooks/{nb}/notes/1").text
    assert 'data-autosave-url="/notebooks/1/notes/1/autosave"' in page
    assert '<script src="/static/notes.js" defer></script>' in page
    ok = c.post(
        f"/notebooks/{nb}/notes/1/autosave",
        data={
            "title": "Linear Algebra Review",
            "content": "Eigen stuff",
            "revision": "1",
        },
        headers=ORIGIN,
    )
    assert ok.json()["ok"] and ok.json()["revision"] == 2
    stale = c.post(
        f"/notebooks/{nb}/notes/1/autosave",
        data={"title": "x", "content": "overwrite", "revision": "1"},
        headers=ORIGIN,
    )
    assert stale.status_code == 409 and stale.json()["conflict"]
    bad = c.post(
        f"/notebooks/{nb}/notes/1/autosave",
        data={"title": "", "content": "x", "revision": "2"},
        headers=ORIGIN,
    )
    assert bad.status_code == 422 and "title" in bad.json()["error"]
    plain = c.post(
        f"/notebooks/{nb}/notes/1",
        data={"title": "LA", "content": "Saved without JS", "revision": "2"},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert plain.status_code == 303
    assert "Saved without JS" in c.get(f"/notebooks/{nb}/notes/1").text
    assert "LA" in c.get(f"/notebooks/{nb}/notes?q=saved").text
    assert "No notes match" in c.get(f"/notebooks/{nb}/notes?q=zzzz").text
    confirm = c.get(f"/notebooks/{nb}/notes/1/delete").text
    assert "Delete this note permanently?" in confirm
    assert c.post(f"/notebooks/{nb}/notes/1/delete", headers=ORIGIN).status_code == 422
    gone = c.post(
        f"/notebooks/{nb}/notes/1/delete",
        data={"confirm": "1"},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert (
        gone.status_code == 303 and c.get(f"/notebooks/{nb}/notes/1").status_code == 404
    )
    assert web["embedder_calls"] == [] and web["llama"] == []


def test_attach_and_add_to_note_flows(web):
    c, nb, ex, chunk = web["c"], web["nb"], web["ex"], web["chunk"]
    c.post(f"/notebooks/{nb}/notes", data={"title": "Collect"}, headers=ORIGIN)
    found = c.get(f"/notebooks/{nb}/notes/1/attach?q=eigenvectors", headers=HX).text
    assert f'name="item_id" value="{chunk[0]}"' in found and "<mark>" in found
    links = c.post(
        f"/notebooks/{nb}/notes/1/links",
        data={"kind": "chunk", "item_id": chunk[0]},
        headers=HX,
    ).text
    assert "e.pdf — page 1" in links and "Eigenvectors and pivots." in links
    form = c.get(
        f"/notebooks/{nb}/notes/add-form?kind=message&id={ex.assistant_message_id}",
        headers=HX,
    ).text
    assert '<option value="new">New note</option>' in form and ">Collect<" in form
    saved = c.post(
        f"/notebooks/{nb}/notes/add",
        data={"kind": "message", "item_id": ex.assistant_message_id, "note_id": "new"},
        headers=HX,
    )
    assert "Saved answer: What are eigenvectors?" in saved.text
    note = c.get(f"/notebooks/{nb}/notes/2").text
    assert "Special vectors [S1]" in note and "[S1] e.pdf — page 1, chunk 0" in note
    assert "Cited: [S1] e.pdf p.1" in note  # citation snapshot kept with the link
    into_existing = c.post(
        f"/notebooks/{nb}/notes/add",
        data={"kind": "chunk", "item_id": chunk[0], "note_id": "2"},
        headers=HX,
    )
    assert into_existing.status_code == 200
    node = re.search(
        r"/memory/nodes/(\d+)", c.get(f"/notebooks/{nb}/memory?q=eigenvectors").text
    ).group(1)
    mem = c.post(
        f"/notebooks/{nb}/notes/add",
        data={"kind": "memory", "item_id": node, "note_id": "1"},
        headers=HX,
    )
    assert mem.status_code == 200
    assert web["embedder_calls"] == [] and web["llama"] == []


def test_note_html_is_escaped_everywhere(web):
    c, nb = web["c"], web["nb"]
    c.post(
        f"/notebooks/{nb}/notes",
        data={"title": '<img src=x onerror="alert(1)">'},
        headers=ORIGIN,
    )
    c.post(
        f"/notebooks/{nb}/notes/1/autosave",
        data={
            "title": '<img src=x onerror="alert(1)">',
            "content": "<script>alert('note')</script>",
            "revision": "1",
        },
        headers=ORIGIN,
    )
    for url in (
        f"/notebooks/{nb}/notes",
        f"/notebooks/{nb}/notes/1",
        f"/notebooks/{nb}/search?q=alert",
        f"/notebooks/{nb}/memory?filter=notes",
    ):
        html = c.get(url).text
        assert "<script>alert(" not in html and "<img src=x" not in html, url
        assert "&lt;" in html, url


def test_cross_notebook_note_access_denied(web):
    c, nb = web["c"], web["nb"]
    c.post(f"/notebooks/{nb}/notes", data={"title": "Private"}, headers=ORIGIN)
    conn = connect(web["s"].db_path)
    other = create_notebook(conn, "Other").id
    conn.close()
    for method, url, data in [
        ("GET", f"/notebooks/{other}/notes/1", None),
        ("GET", f"/notebooks/{other}/notes/1/delete", None),
        ("POST", f"/notebooks/{other}/notes/1/delete", {"confirm": "1"}),
        (
            "POST",
            f"/notebooks/{other}/notes/1/autosave",
            {"title": "hijack", "content": "x", "revision": "1"},
        ),
        ("POST", f"/notebooks/{other}/notes/1", {"title": "hijack"}),
        ("GET", f"/notebooks/{other}/notes/1/attach?q=x", None),
        (
            "POST",
            f"/notebooks/{other}/notes/1/links",
            {"kind": "chunk", "item_id": "1"},
        ),
        (
            "POST",
            f"/notebooks/{other}/notes/add",
            {"kind": "chunk", "item_id": web["chunk"][0], "note_id": "1"},
        ),
    ]:
        assert c.request(method, url, data=data, headers=HX).status_code == 404, url
    assert (
        c.post(
            f"/notebooks/{nb}/notes/1/autosave",
            data={"title": "x"},
            headers={"Origin": "http://evil"},
        ).status_code
        == 403
    )
    assert "Private" in c.get(f"/notebooks/{nb}/notes").text


def test_migration_0007_preserves_populated_schema6_db(settings, tmp_path, monkeypatch):
    import shutil

    from litenotebook import db
    from litenotebook.db import available_migrations, migrate
    from litenotebook.services.settings import SettingsService

    partial = tmp_path / "m6"
    partial.mkdir()
    for _v, path in available_migrations()[:6]:
        shutil.copy(path, partial / path.name)
    conn = connect(settings.db_path)
    with monkeypatch.context() as m:
        m.setattr(db, "MIGRATIONS_DIR", partial)
        assert migrate(conn) == 6
    nb = create_notebook(conn, "Maths")
    store = FaissStore(conn, settings)
    ingest_pdf(
        conn,
        settings,
        nb,
        make_pdf(tmp_path / "m.pdf", ["Pivot rank facts."]),
        FakeEmbedder(),
        store,
    )
    ex = start_exchange(conn, nb.id, None, "What is rank?")
    conn.execute(
        "UPDATE messages SET status='complete', content='Pivots.', model='q' "
        "WHERE id = ?",
        (ex.assistant_message_id,),
    )
    conn.commit()
    # Phase 6-era graph rows and outcome, as the old code wrote them
    cid, aid, uid = ex.conversation.id, ex.assistant_message_id, ex.user_message_id
    rows = [
        (
            1,
            f"conversation:{cid}",
            "conversation",
            "What is rank?",
            "derived",
            cid,
            None,
        ),
        (2, f"question:{uid}", "question", "What is rank?", "user_statement", cid, uid),
        (3, f"answer:{aid}", "answer", "Pivots.", "assistant_response", cid, aid),
        (4, "concept:rank", "concept", "rank", "derived", None, None),
    ]
    conn.executemany(
        "INSERT INTO memory_nodes (id, notebook_id, key, node_type, label, content, "
        "provenance, conversation_id, message_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(i, nb.id, k, t, lab, lab, pr, c, m) for i, k, t, lab, pr, c, m in rows],
    )
    conn.executemany(
        "INSERT INTO memory_edges (notebook_id, from_node_id, to_node_id, relation, "
        "provenance) VALUES (?, ?, ?, ?, ?)",
        [
            (nb.id, 1, 2, "contains", f"message:{aid}"),
            (nb.id, 1, 3, "contains", f"message:{aid}"),
            (nb.id, 2, 3, "answered_by", f"message:{aid}"),
            (nb.id, 2, 4, "mentions", f"message:{aid}"),
        ],
    )
    conn.execute(
        "INSERT INTO memory_outcomes (notebook_id, message_id, outcome) "
        "VALUES (?, ?, 'useful')",
        (nb.id, aid),
    )
    conn.commit()
    SettingsService(conn, env={}).update(rag_top_k=6)
    tables = (
        "notebooks",
        "sources",
        "chunks",
        "conversations",
        "messages",
        "memory_nodes",
        "memory_edges",
        "memory_outcomes",
        "settings",
    )
    before = {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables
    }
    node_rows = conn.execute("SELECT id, key FROM memory_nodes ORDER BY id").fetchall()

    assert migrate(conn) == len(available_migrations())
    after = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    assert after == before
    assert (
        conn.execute("SELECT id, key FROM memory_nodes ORDER BY id").fetchall()
        == node_rows
    )
    assert store.check(nb.id).ok
    edges = conn.execute(
        "SELECT from_node_id, to_node_id, relation FROM memory_edges ORDER BY id"
    ).fetchall()
    assert [tuple(e) for e in edges] == [
        (1, 2, "contains"),
        (1, 3, "contains"),
        (2, 3, "answered_by"),
        (2, 4, "mentions"),
    ]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    hits = conn.execute(
        "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'pivots'"
    ).fetchall()
    assert [h[0] for h in hits] == [ex.assistant_message_id]  # backfilled index
    assert search_memory(conn, nb.id, "rank").items  # memory FTS re-attached
    create_note(conn, nb.id, "Works after upgrade", "rank")
    conn.close()
