import json

import faiss
import numpy as np
import pytest

from conftest import make_pdf
from litenotebook.index.faiss_store import IndexMismatchError, index_ids, new_index
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.notebooks import create_notebook

PAGES = [
    "Gaussian elimination reduces a matrix to row echelon form using pivots.",
    "The rank of a matrix equals the number of pivot columns.",
    "A homogeneous system always has the trivial solution.",
]


@pytest.fixture
def ingested(conn, settings, notebook, embedder, store, tmp_path):
    ingest_pdf(
        conn, settings, notebook, make_pdf(tmp_path / "la.pdf", PAGES), embedder, store
    )
    return notebook


def chunk_ids(conn, notebook_id):
    return [
        r[0]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE notebook_id = ? ORDER BY id", (notebook_id,)
        )
    ]


def test_index_is_idmap2_keyed_by_chunk_id(ingested, conn, store):
    index = store.load(ingested.id)
    assert isinstance(index, faiss.IndexIDMap2) and index.d == 384
    assert list(index_ids(index)) == chunk_ids(conn, ingested.id)
    assert store.check(ingested.id).ok


def test_search_returns_chunk_ids(ingested, conn, store, embedder):
    query = embedder.encode(["rank pivot columns"])[0]
    hits = store.search(ingested.id, query, 3)
    best = conn.execute(
        "SELECT text FROM chunks WHERE id = ?", (hits[0][0],)
    ).fetchone()
    assert "rank" in best[0]
    assert all(-1.0001 <= s <= 1.0001 for _, s in hits)


def test_save_load_roundtrip_and_metadata(ingested, store):
    meta = json.loads(store.meta_path(ingested.id).read_text())
    assert meta["embed_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta["embed_dim"] == 384 and meta["ntotal"] == 3
    reloaded = faiss.read_index(str(store.index_path(ingested.id)))
    assert reloaded.ntotal == 3


def test_atomic_writes_leave_no_temp_files(ingested, store):
    store.rebuild(ingested.id)
    store.rebuild(ingested.id)
    leftovers = list(store.index_path(ingested.id).parent.glob("*.tmp"))
    assert leftovers == []


def test_missing_index_detected_and_rebuilt_from_sqlite(ingested, conn, store):
    store.index_path(ingested.id).unlink()
    status = store.check(ingested.id)
    assert not status.ok and status.needs_rebuild and "missing" in status.problems[0]
    with pytest.raises(IndexMismatchError):
        store.load(ingested.id, repair=False)
    index = store.load(ingested.id)  # repairs
    assert list(index_ids(index)) == chunk_ids(conn, ingested.id)
    assert store.check(ingested.id).ok


def test_rebuild_reproduces_identical_vectors(ingested, conn, store, embedder):
    before = store.load(ingested.id)
    query = embedder.encode(["homogeneous trivial solution"])[0].reshape(1, -1)
    expected = before.search(query, 3)
    after = store.rebuild(ingested.id)
    got = after.search(query, 3)
    assert np.array_equal(expected[1], got[1]) and np.allclose(expected[0], got[0])


def test_count_mismatch_detected_and_synced(ingested, conn, store):
    removed = chunk_ids(conn, ingested.id)[0]
    conn.execute("DELETE FROM chunks WHERE id = ?", (removed,))
    status = store.check(ingested.id)
    assert any("count mismatch" in p for p in status.problems)
    assert not status.needs_rebuild
    result = store.sync(ingested.id)
    assert (result.added, result.removed, result.rebuilt) == (0, 1, False)
    assert removed not in index_ids(store.load(ingested.id))


def test_id_mismatch_detected(ingested, conn, store):
    index = store.load(ingested.id)
    first = int(index_ids(index)[0])
    vec = index.reconstruct(first).reshape(1, -1)
    index.remove_ids(np.array([first], dtype=np.int64))
    index.add_with_ids(vec, np.array([999_999], dtype=np.int64))
    store._save(ingested.id, index)
    assert any("id mismatch" in p for p in store.check(ingested.id).problems)
    store.sync(ingested.id)
    assert store.check(ingested.id).ok


def test_wrong_model_in_metadata_forces_rebuild(ingested, store):
    meta_path = store.meta_path(ingested.id)
    meta = json.loads(meta_path.read_text())
    meta["embed_model"] = "Qwen/Qwen3-4B"
    meta_path.write_text(json.dumps(meta))
    status = store.check(ingested.id)
    assert status.needs_rebuild and any("Qwen" in p for p in status.problems)
    assert store.sync(ingested.id).rebuilt
    assert store.check(ingested.id).ok


def test_wrong_dimension_index_forces_rebuild(ingested, store):
    faiss.write_index(new_index(768), str(store.index_path(ingested.id)))
    status = store.check(ingested.id)
    assert status.needs_rebuild and any("dimension" in p for p in status.problems)
    assert store.load(ingested.id).d == 384


def test_stale_metadata_detected(ingested, store):
    meta_path = store.meta_path(ingested.id)
    meta = json.loads(meta_path.read_text())
    meta["ntotal"] = 42
    meta_path.write_text(json.dumps(meta))
    assert any("stale" in p for p in store.check(ingested.id).problems)


def test_corrupt_index_file_is_rebuilt(ingested, store):
    store.index_path(ingested.id).write_bytes(b"garbage")
    status = store.check(ingested.id)
    assert status.needs_rebuild and "unreadable" in status.problems[0]
    assert store.load(ingested.id).ntotal == 3


def test_notebook_with_foreign_embedding_model_cannot_be_rebuilt(conn, store):
    conn.execute(
        "INSERT INTO notebooks (id, name, embed_model, embed_dim) "
        "VALUES (50, 'old', 'x', 768)"
    )
    assert any(
        "re-embedding" in p or "embeds with" in p for p in store.check(50).problems
    )
    with pytest.raises(IndexMismatchError):
        store.rebuild(50)


def test_indexes_are_isolated_per_notebook(conn, settings, embedder, store, tmp_path):
    a, b = create_notebook(conn, "a"), create_notebook(conn, "b")
    ingest_pdf(
        conn, settings, a, make_pdf(tmp_path / "a.pdf", PAGES[:1]), embedder, store
    )
    ingest_pdf(
        conn, settings, b, make_pdf(tmp_path / "b.pdf", PAGES[1:]), embedder, store
    )
    assert store.index_path(a.id) != store.index_path(b.id)
    assert set(index_ids(store.load(a.id))) == set(chunk_ids(conn, a.id))
    assert set(index_ids(store.load(b.id))) == set(chunk_ids(conn, b.id))
    assert not set(chunk_ids(conn, a.id)) & set(chunk_ids(conn, b.id))


def test_new_empty_notebook_without_index_is_ok(conn, store, notebook):
    status = store.check(notebook.id)
    assert status.ok and status.sqlite_count == 0 and status.index_count is None
    assert store.load(notebook.id).ntotal == 0  # still loads (built on demand)


def test_chunk_ids_are_never_reused(
    conn, settings, notebook, embedder, store, tmp_path
):
    ingest_pdf(
        conn, settings, notebook, make_pdf(tmp_path / "x.pdf", PAGES), embedder, store
    )
    old = chunk_ids(conn, notebook.id)
    conn.execute("DELETE FROM sources")
    ingest_pdf(
        conn, settings, notebook, make_pdf(tmp_path / "x.pdf", PAGES), embedder, store
    )
    assert min(chunk_ids(conn, notebook.id)) > max(old)
