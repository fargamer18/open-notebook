"""Phase 3 source lifecycle: register → process → ready/failed → retry/delete."""

import json

import numpy as np
import pytest

from conftest import FakeEmbedder, make_pdf
from litenotebook.ingest import pipeline
from litenotebook.ingest.pipeline import (
    DuplicateSourceError,
    IngestError,
    UnsupportedFileError,
    delete_source,
    process_source,
    register_source,
)
from litenotebook.notebooks import create_notebook, delete_notebook
from litenotebook.retrieval.hybrid import Retriever
from litenotebook.sources import (
    INTERRUPTED_MESSAGE,
    SourceBusyError,
    SourceError,
    SourceNotFoundError,
    display_filename,
    get_source,
    list_sources,
    recover_interrupted,
    reset_for_retry,
)

PAGES = [
    "Gaussian elimination reduces a matrix to row echelon form using pivots.",
    "The rank of a matrix equals the number of pivots in its echelon form.",
    "Zorbania's capital is Quillport, a fictional harbour city.",
]


@pytest.fixture
def pdf(tmp_path):
    return make_pdf(tmp_path / "linear.pdf", PAGES)


def counts(conn, source_id):
    return {
        table: conn.execute(
            f"SELECT count(*) FROM {table} WHERE source_id = ?", (source_id,)
        ).fetchone()[0]
        for table in ("pages", "chunks")
    }


def fts_hits(conn, word):
    return conn.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (word,)
    ).fetchone()[0]


class FailingEmbedder(FakeEmbedder):
    def encode(self, texts):
        raise RuntimeError("simulated crash /home/secret/path")


# --- display filenames -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\evil.pdf", "evil.pdf"),
        ("/abs/path/doc.pdf", "doc.pdf"),
        ("a\x00b\x1fc.pdf", "abc.pdf"),
        ("‮fdp.exe", "fdp.exe"),  # RTL override (Cf) dropped
        ("..", "document.pdf"),
        ("", "document.pdf"),
        ("  .hidden.pdf ", "hidden.pdf"),
        ("x" * 300 + ".pdf", "x" * 196 + ".pdf"),
        ("Café.pdf", "Café.pdf"),  # NFC
    ],
)
def test_display_filename(raw, expected):
    assert display_filename(raw) == expected


# --- register ----------------------------------------------------------------------


def test_register_stores_by_id_and_records_pending(conn, settings, notebook, pdf):
    src = register_source(conn, settings, notebook, pdf, filename="../../x/<b>a<i>.pdf")
    assert src.status == "pending" and src.stage == "uploaded"
    assert src.original_filename == "<b>a<i>.pdf" and src.page_count == 3
    stored = settings.data_dir / src.stored_path
    assert stored == settings.source_file(notebook.id, src.id)
    assert stored.name == f"{src.id}.pdf" and stored.read_bytes() == pdf.read_bytes()
    assert pdf.exists()  # copied, not moved
    assert counts(conn, src.id) == {"pages": 0, "chunks": 0}


def test_register_move_consumes_input(conn, settings, notebook, pdf):
    src = register_source(conn, settings, notebook, pdf, move=True)
    assert not pdf.exists()
    assert (settings.data_dir / src.stored_path).is_file()


@pytest.mark.parametrize(
    "payload",
    [b"MZ\x90\x00 an executable", b"<html><script>alert(1)</script>", b"\x00" * 50],
)
def test_register_rejects_non_pdf_content(conn, settings, notebook, tmp_path, payload):
    fake = tmp_path / "looks-like.pdf"
    fake.write_bytes(payload)
    with pytest.raises(UnsupportedFileError, match="not a PDF"):
        register_source(conn, settings, notebook, fake)
    assert list_sources(conn, notebook.id) == []
    assert not settings.sources_dir(notebook.id).exists() or not any(
        settings.sources_dir(notebook.id).iterdir()
    )


def test_register_rejects_damaged_pdf(conn, settings, notebook, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\n" + b"garbage" * 100)
    with pytest.raises(IngestError, match="damaged or not a valid PDF|no pages"):
        register_source(conn, settings, notebook, broken)
    assert list_sources(conn, notebook.id) == []


def test_real_pdf_with_misleading_name_is_accepted(conn, settings, notebook, pdf):
    src = register_source(conn, settings, notebook, pdf, filename="notes.txt")
    assert src.original_filename == "notes.txt" and src.mime == "application/pdf"
    assert (settings.data_dir / src.stored_path).suffix == ".pdf"


def test_duplicate_in_same_notebook_rejected(conn, settings, notebook, pdf, tmp_path):
    first = register_source(conn, settings, notebook, pdf)
    copy = tmp_path / "renamed.pdf"
    copy.write_bytes(pdf.read_bytes())
    with pytest.raises(DuplicateSourceError) as exc:
        register_source(conn, settings, notebook, copy, filename="renamed.pdf")
    assert exc.value.source_id == first.id
    assert "already in this notebook" in str(exc.value) and "linear.pdf" in str(
        exc.value
    )
    assert len(list_sources(conn, notebook.id)) == 1


def test_same_pdf_allowed_in_different_notebooks(
    conn, settings, notebook, pdf, embedder, store
):
    other = create_notebook(conn, "other")
    a = register_source(conn, settings, notebook, pdf)
    b = register_source(conn, settings, other, pdf)
    assert a.sha256 == b.sha256 and a.stored_path != b.stored_path
    process_source(conn, settings, a.id, embedder, store)
    process_source(conn, settings, b.id, embedder, store)
    assert store.check(notebook.id).ok and store.check(other.id).ok
    delete_source(conn, settings, notebook.id, a.id, store)
    assert get_source(conn, other.id, b.id).status == "ready"
    assert (settings.data_dir / b.stored_path).is_file()


# --- process -----------------------------------------------------------------------


def test_successful_ingestion_and_status_transitions(
    conn, settings, notebook, pdf, embedder, store
):
    src = register_source(conn, settings, notebook, pdf)
    seen = []
    report = process_source(
        conn,
        settings,
        src.id,
        embedder,
        store,
        on_progress=lambda stage, done, total: seen.append((stage, done, total)),
    )
    stages = [s for s, _, _ in seen]
    assert stages[0] == "extracting" and stages[-1] == "indexing"
    assert (
        stages.index("chunking") < stages.index("embedding") < stages.index("indexing")
    )
    assert ("extracting", 3, 3) in seen and (
        "embedding",
        report.chunks,
        report.chunks,
    ) in seen

    ready = get_source(conn, notebook.id, src.id)
    assert ready.status == "ready" and ready.stage == "ready" and ready.error is None
    assert ready.chunk_count == report.chunks == 3 and ready.page_count == 3
    assert ready.attempts == 1 and ready.started_at and ready.finished_at
    assert {"extract", "chunk", "embed", "sqlite", "faiss", "total"} <= set(
        ready.timings
    )
    assert store.check(notebook.id).ok
    assert fts_hits(conn, "Quillport") == 1


def test_model_load_time_is_reported_separately(conn, settings, notebook, pdf, store):
    class LoadingEmbedder(FakeEmbedder):
        loads = 0

        def load(self):
            LoadingEmbedder.loads += 1

    src = register_source(conn, settings, notebook, pdf)
    report = process_source(conn, settings, src.id, LoadingEmbedder(), store)
    assert LoadingEmbedder.loads == 1 and "model_load" in report.timings
    stored = json.loads(
        conn.execute("SELECT timings FROM sources WHERE id = ?", (src.id,)).fetchone()[
            0
        ]
    )
    assert "model_load" in stored and "embed" in stored


def test_progress_is_visible_from_another_connection(
    conn, settings, notebook, pdf, embedder, store
):
    from litenotebook.db import connect

    src = register_source(conn, settings, notebook, pdf)
    other = connect(settings.db_path)
    snapshots = []

    def hook(stage, done, total):
        row = other.execute(
            "SELECT status, stage, progress_done, progress_total FROM sources "
            "WHERE id = ?",
            (src.id,),
        ).fetchone()
        snapshots.append(tuple(row))

    process_source(conn, settings, src.id, embedder, store, on_progress=hook)
    other.close()
    assert ("processing", "extracting", 3, 3) in snapshots
    assert any(s[1] == "embedding" and s[2] == s[3] for s in snapshots)
    # derived rows are never visible before the source is ready
    assert all(s[0] == "processing" for s in snapshots)


def test_only_pending_sources_are_processed(
    conn, settings, notebook, pdf, embedder, store
):
    src = register_source(conn, settings, notebook, pdf)
    process_source(conn, settings, src.id, embedder, store)
    with pytest.raises(IngestError, match="not pending"):
        process_source(conn, settings, src.id, embedder, store)
    with pytest.raises(IngestError, match="no longer exists"):
        process_source(conn, settings, 9999, embedder, store)


def test_failed_ingestion_leaves_no_partial_data(
    conn, settings, notebook, pdf, store, caplog
):
    src = register_source(conn, settings, notebook, pdf)
    with pytest.raises(IngestError) as exc:
        process_source(conn, settings, src.id, FailingEmbedder(), store)
    message = str(exc.value)
    assert "Unexpected error while embedding" in message
    assert "/home/secret" not in message and "Traceback" not in message
    failed = get_source(conn, notebook.id, src.id)
    assert failed.status == "failed" and failed.stage == "failed"
    assert failed.error == message and failed.finished_at
    assert counts(conn, src.id) == {"pages": 0, "chunks": 0}
    assert store.check(notebook.id).ok
    assert (settings.data_dir / src.stored_path).is_file()  # kept for retry
    assert "simulated crash" in caplog.text  # full detail goes to the log only


def test_faiss_failure_after_commit_rolls_back_to_failed(
    conn, settings, notebook, pdf, embedder, store, monkeypatch
):
    good = register_source(conn, settings, notebook, pdf)
    process_source(conn, settings, good.id, embedder, store)
    other_pdf = make_pdf(pdf.with_name("second.pdf"), ["Eigenvalues of a matrix."])
    src = register_source(conn, settings, notebook, other_pdf)

    real_sync, calls = store.sync, []

    def flaky_sync(notebook_id):
        calls.append(notebook_id)
        if len(calls) == 1:
            raise OSError("disk full")
        return real_sync(notebook_id)

    monkeypatch.setattr(store, "sync", flaky_sync)
    with pytest.raises(IngestError, match="Unexpected error while indexing"):
        process_source(conn, settings, src.id, embedder, store)
    assert get_source(conn, notebook.id, src.id).status == "failed"
    assert counts(conn, src.id) == {"pages": 0, "chunks": 0}
    assert fts_hits(conn, "Eigenvalues") == 0
    monkeypatch.setattr(store, "sync", real_sync)
    status = store.check(notebook.id)
    assert (
        status.ok
        and status.sqlite_count == get_source(conn, notebook.id, good.id).chunk_count
    )


def test_scanned_pdf_without_ocr_fails_with_hint(
    conn, settings, notebook, tmp_path, embedder, store
):
    scan = make_pdf(tmp_path / "scan.pdf", ["SCANNED TEXT"], image_only={1})
    src = register_source(conn, settings, notebook, scan)
    with pytest.raises(IngestError, match="No extractable text"):
        process_source(conn, settings, src.id, embedder, store)
    failed = get_source(conn, notebook.id, src.id)
    assert "OCR" in failed.error and failed.page_count == 1


def test_ocr_option_propagates_to_extraction(
    conn, settings, notebook, pdf, embedder, store, monkeypatch
):
    seen = []
    real = pipeline.extract_pages

    def spy(path, **kw):
        seen.append(kw["ocr"])
        return real(path, **kw)

    monkeypatch.setattr(pipeline, "extract_pages", spy)
    on = register_source(conn, settings, notebook, pdf, ocr=True)
    assert on.ocr is True
    process_source(conn, settings, on.id, embedder, store)
    off_pdf = make_pdf(pdf.with_name("other.pdf"), ["Different text entirely."])
    off = register_source(conn, settings, notebook, off_pdf)
    assert off.ocr is False
    process_source(conn, settings, off.id, embedder, store)
    assert seen == [True, False]


# --- retry -------------------------------------------------------------------------


def test_retry_reuses_record_and_file(conn, settings, notebook, pdf, embedder, store):
    src = register_source(conn, settings, notebook, pdf)
    with pytest.raises(IngestError):
        process_source(conn, settings, src.id, FailingEmbedder(), store)
    stored = (settings.data_dir / src.stored_path).read_bytes()

    pending = reset_for_retry(conn, settings, notebook.id, src.id)
    assert (
        pending.id == src.id and pending.status == "pending" and pending.error is None
    )
    process_source(conn, settings, src.id, embedder, store)

    ready = get_source(conn, notebook.id, src.id)
    assert ready.status == "ready" and ready.attempts == 2 and ready.chunk_count == 3
    assert len(list_sources(conn, notebook.id)) == 1
    assert (settings.data_dir / src.stored_path).read_bytes() == stored
    assert store.check(notebook.id).ok


def test_retry_only_for_failed_and_scoped_to_notebook(
    conn, settings, notebook, pdf, embedder, store
):
    other = create_notebook(conn, "other")
    src = register_source(conn, settings, notebook, pdf)
    with pytest.raises(SourceBusyError):
        reset_for_retry(conn, settings, notebook.id, src.id)  # pending
    with pytest.raises(IngestError):
        process_source(conn, settings, src.id, FailingEmbedder(), store)
    with pytest.raises(SourceNotFoundError):
        reset_for_retry(conn, settings, other.id, src.id)
    assert get_source(conn, notebook.id, src.id).status == "failed"


def test_retry_with_missing_file_is_refused(
    conn, settings, notebook, pdf, embedder, store
):
    src = register_source(conn, settings, notebook, pdf)
    with pytest.raises(IngestError):
        process_source(conn, settings, src.id, FailingEmbedder(), store)
    (settings.data_dir / src.stored_path).unlink()
    with pytest.raises(SourceError, match="missing"):
        reset_for_retry(conn, settings, notebook.id, src.id)


# --- restart recovery --------------------------------------------------------------


def test_recover_interrupted_marks_failed_without_rerunning(
    conn, settings, notebook, pdf, tmp_path
):
    pending = register_source(conn, settings, notebook, pdf)
    busy_pdf = make_pdf(tmp_path / "busy.pdf", ["Some text on a page."])
    busy = register_source(conn, settings, notebook, busy_pdf)
    conn.execute(
        "UPDATE sources SET status = 'processing', stage = 'embedding' WHERE id = ?",
        (busy.id,),
    )
    conn.commit()
    assert sorted(recover_interrupted(conn)) == sorted([pending.id, busy.id])
    for sid in (pending.id, busy.id):
        src = get_source(conn, notebook.id, sid)
        assert src.status == "failed" and src.error == INTERRUPTED_MESSAGE
    assert recover_interrupted(conn) == []


# --- delete ------------------------------------------------------------------------


def test_delete_source_removes_everything(
    conn, settings, notebook, pdf, embedder, store, tmp_path
):
    keep_pdf = make_pdf(tmp_path / "keep.pdf", ["Determinants and cofactors."])
    keep = register_source(conn, settings, notebook, keep_pdf)
    gone = register_source(conn, settings, notebook, pdf)
    for sid in (keep.id, gone.id):
        process_source(conn, settings, sid, embedder, store)
    stored = settings.data_dir / gone.stored_path
    assert fts_hits(conn, "Quillport") == 1

    deleted = delete_source(conn, settings, notebook.id, gone.id, store)
    assert deleted.original_filename == "linear.pdf"
    with pytest.raises(SourceNotFoundError):
        get_source(conn, notebook.id, gone.id)
    assert counts(conn, gone.id) == {"pages": 0, "chunks": 0}
    assert fts_hits(conn, "Quillport") == 0 and fts_hits(conn, "cofactors") == 1
    assert not stored.exists() and (settings.data_dir / keep.stored_path).exists()
    status = store.check(notebook.id)
    assert (
        status.ok
        and status.index_count == get_source(conn, notebook.id, keep.id).chunk_count
    )
    results = Retriever(conn, settings, embedder, store).search(
        notebook.id, "Quillport"
    )
    assert all(r.source_id == keep.id for r in results)


def test_delete_is_notebook_scoped_and_refuses_processing(
    conn, settings, notebook, pdf, store
):
    other = create_notebook(conn, "other")
    src = register_source(conn, settings, notebook, pdf)
    with pytest.raises(SourceNotFoundError):
        delete_source(conn, settings, other.id, src.id, store)
    conn.execute("UPDATE sources SET status = 'processing' WHERE id = ?", (src.id,))
    conn.commit()
    with pytest.raises(SourceBusyError):
        delete_source(conn, settings, notebook.id, src.id, store)
    assert (settings.data_dir / src.stored_path).is_file()


def test_delete_never_touches_paths_outside_sources_dir(
    conn, settings, notebook, pdf, store, tmp_path
):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    src = register_source(conn, settings, notebook, pdf)
    for evil in ("../../../victim.txt", str(victim), "notebooks/1/../../../victim.txt"):
        conn.execute("UPDATE sources SET stored_path = ? WHERE id = ?", (evil, src.id))
        conn.commit()
        assert get_source(conn, notebook.id, src.id).stored_path == evil
    delete_source(conn, settings, notebook.id, src.id, store)
    assert victim.read_text() == "precious"


def test_delete_unlinks_symlink_target_never(
    conn, settings, notebook, pdf, store, tmp_path
):
    victim = tmp_path / "victim.pdf"
    victim.write_bytes(pdf.read_bytes())
    src = register_source(conn, settings, notebook, pdf)
    stored = settings.data_dir / src.stored_path
    stored.unlink()
    stored.symlink_to(victim)
    delete_source(conn, settings, notebook.id, src.id, store)
    assert victim.exists()


def test_notebook_delete_cascades_sources(
    conn, settings, notebook, pdf, embedder, store
):
    src = register_source(conn, settings, notebook, pdf)
    process_source(conn, settings, src.id, embedder, store)
    result = delete_notebook(conn, settings, notebook.id)
    assert result.sources == 1 and result.chunks == 3
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert fts_hits(conn, "Quillport") == 0
    assert not settings.notebook_dir(notebook.id).exists()


def test_notebook_delete_refused_while_ingesting(conn, settings, notebook, pdf):
    from litenotebook.notebooks import NotebookError

    src = register_source(conn, settings, notebook, pdf)
    conn.execute("UPDATE sources SET status = 'processing' WHERE id = ?", (src.id,))
    conn.commit()
    with pytest.raises(NotebookError, match="being ingested"):
        delete_notebook(conn, settings, notebook.id)


def test_notebook_isolation_of_listing(conn, settings, notebook, pdf, tmp_path):
    other = create_notebook(conn, "other")
    register_source(conn, settings, notebook, pdf)
    register_source(
        conn, settings, other, make_pdf(tmp_path / "o.pdf", ["Other notebook text."])
    )
    assert [s.original_filename for s in list_sources(conn, notebook.id)] == [
        "linear.pdf"
    ]
    assert [s.original_filename for s in list_sources(conn, other.id)] == ["o.pdf"]


def test_cli_ingest_still_all_or_nothing(conn, settings, notebook, pdf, store):
    with pytest.raises(IngestError):
        pipeline.ingest_pdf(conn, settings, notebook, pdf, FailingEmbedder(), store)
    assert list_sources(conn, notebook.id) == []
    assert not any(settings.sources_dir(notebook.id).glob("*"))
    assert np.all(store.sqlite_ids(notebook.id) == np.array([], dtype=np.int64))
