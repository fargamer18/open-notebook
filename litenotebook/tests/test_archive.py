"""Phase 10: export, import, full backup, verify and restore."""

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import tracemalloc
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from archive_fixture import (
    build_rich_notebook,
    edit_rows,
    logical_snapshot,
    read_entries,
    write_archive,
)
from conftest import FakeEmbedder, make_pdf
from litenotebook.archive.export import create_backup, export_notebook, summary
from litenotebook.archive.format import (
    FORMAT_VERSION,
    NEWER_VERSION_MESSAGE,
    TABLES,
    ArchiveError,
    Limits,
    UnsupportedArchiveError,
)
from litenotebook.archive.importer import (
    ImportFailedError,
    import_archive,
    import_notebook,
    open_and_verify,
    remap_key,
    unique_name,
)
from litenotebook.archive.integrity import database_report, notebook_report
from litenotebook.archive.reader import verify_archive
from litenotebook.archive.restore import (
    MARKER,
    RestoreBusyError,
    apply_pending_restore,
    cancel_restore,
    pending_restore,
    preview,
    stage_restore,
    staging_dir,
)
from litenotebook.config import EMBED_MODEL, load_settings
from litenotebook.db import available_migrations, open_db, schema_version
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source, ingest_pdf
from litenotebook.notebooks import create_notebook, delete_notebook
from litenotebook.platform_utils.filesystem import InstanceLock
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
ORIGIN = {"Origin": BASE}
SECRET = "sk-test-SECRET-123456"

CURRENT_SCHEMA = len(available_migrations())


@pytest.fixture
def rich(conn, settings, tmp_path):
    return build_rich_notebook(conn, settings, tmp_path)


def limits(settings):
    return Limits.from_settings(settings)


def verify(path, settings, conn=None):
    return verify_archive(
        path,
        limits(settings),
        current_schema=schema_version(conn) if conn else CURRENT_SCHEMA,
    )


def notebook_files(settings):
    root = settings.notebooks_dir
    return (
        sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
        if root.exists()
        else []
    )


# --- export -------------------------------------------------------------------------


def test_export_layout_manifest_and_checksums(conn, settings, tmp_path, rich):
    result = export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip")
    entries = read_entries(result.path)
    nb = f"notebooks/{rich.notebook_id}"
    expected = {"manifest.json", "checksums.json", f"{nb}/notebook.json"}
    expected |= {f"{nb}/tables/{t}.jsonl" for t in TABLES}
    expected |= {f"{nb}/sources/{s}.pdf" for s in rich.sources}
    assert set(entries) == expected
    manifest = json.loads(entries["manifest.json"])
    assert manifest["format"] == "litenotebook-archive"
    assert manifest["export_format_version"] == FORMAT_VERSION == 2  # Phase 13
    assert (
        manifest["kind"] == "notebook" and manifest["schema_version"] == CURRENT_SCHEMA
    )
    assert manifest["embedding"] == {
        "model": EMBED_MODEL,
        "dim": 384,
        "dtype": "float32-le",
        "normalized": True,
    }
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", manifest["created_at"])
    c = manifest["counts"]
    assert (
        c["sources"],
        c["chunks"],
        c["embedded_chunks"],
        c["notes"],
        c["conversations"],
        c["messages"],
        c["annotations"],
    ) == (2, 5, 5, 1, 1, 4, 1)
    assert c["message_citations"] == 1 and c["message_memory_refs"] == 1
    assert (
        c["memory_outcomes"] == 1 and c["note_links"] == 4 and c["reading_state"] == 1
    )
    checksums = json.loads(entries["checksums.json"])["files"]
    assert (
        hashlib.sha256(entries["checksums.json"]).hexdigest()
        == manifest["checksums_sha256"]
    )
    for name, blob in entries.items():
        if name not in ("manifest.json", "checksums.json"):
            assert hashlib.sha256(blob).hexdigest() == checksums[name], name
    for sid, sha in conn.execute(
        "SELECT id, sha256 FROM sources WHERE notebook_id = ?", (rich.notebook_id,)
    ):
        assert checksums[f"{nb}/sources/{sid}.pdf"] == sha
    assert hashlib.sha256(result.path.read_bytes()).hexdigest() == result.sha256
    with zipfile.ZipFile(result.path) as zf:  # PDFs stored, JSON compressed
        kinds = {i.filename.rsplit(".", 1)[1]: i.compress_type for i in zf.infolist()}
    assert kinds["pdf"] == zipfile.ZIP_STORED and kinds["jsonl"] == zipfile.ZIP_DEFLATED


def test_export_contains_no_secrets_paths_or_derived_files(
    conn, settings, tmp_path, rich
):
    conn.execute(
        "INSERT INTO settings (key, value) VALUES ('llm.api_key', ?)", (SECRET,)
    )
    conn.commit()
    from litenotebook.reader.annotations import open_source
    from litenotebook.reader.render import render_page

    render_page(
        settings, open_source(conn, settings, rich.notebook_id, rich.sources[0]), 1, 1.0
    )
    assert settings.faiss_index_path(rich.notebook_id).exists()
    for result in (
        export_notebook(conn, settings, rich.notebook_id, tmp_path / "n.zip"),
        create_backup(conn, settings, tmp_path / "b.zip"),
    ):
        raw = result.path.read_bytes()
        entries = read_entries(result.path)
        everything = raw + b"".join(entries.values())
        assert SECRET.encode() not in everything
        assert str(settings.data_dir).encode() not in everything
        assert str(tmp_path).encode() not in everything
        assert b"stored_path" not in everything and b"index.faiss" not in everything
        assert not any(
            n.endswith((".faiss", ".png", ".json.tmp")) or "cache" in n for n in entries
        )
    assert "settings.json" not in read_entries(tmp_path / "n.zip")
    backup_settings = json.loads(read_entries(tmp_path / "b.zip")["settings.json"])
    assert {r["key"] for r in backup_settings["settings"]} <= {
        "llm.base_url",
        "llm.model",
        "rag.top_k",
    }


def test_export_refuses_busy_or_missing_sources(conn, settings, tmp_path, rich):
    conn.execute(
        "UPDATE sources SET status = 'processing' WHERE id = ?", (rich.sources[1],)
    )
    conn.commit()
    with pytest.raises(ArchiveError, match="still being processed"):
        export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip")
    assert summary(conn, rich.notebook_id)["busy"] == 1
    conn.execute("UPDATE sources SET status = 'ready' WHERE id = ?", (rich.sources[1],))
    conn.commit()
    settings.source_file(rich.notebook_id, rich.sources[1]).unlink()
    with pytest.raises(ArchiveError, match="missing"):
        export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip")
    assert not list(tmp_path.glob("*.partial")) and not (tmp_path / "x.zip").exists()


def test_export_detects_a_corrupted_stored_pdf(conn, settings, tmp_path, rich):
    path = settings.source_file(rich.notebook_id, rich.sources[0])
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ArchiveError, match="checksum"):
        export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip")


# --- import -------------------------------------------------------------------------


def test_round_trip_preserves_everything_with_new_ids(conn, settings, tmp_path, rich):
    before = logical_snapshot(conn, rich.notebook_id)
    out = export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip").path
    delete_notebook(conn, settings, rich.notebook_id)
    filler = create_notebook(conn, "Filler")  # shifts every id
    ingest_pdf(
        conn,
        settings,
        filler,
        make_pdf(tmp_path / "f.pdf", ["Other text."]),
        FakeEmbedder(),
        FaissStore(conn, settings),
    )
    report, [result] = import_archive(conn, settings, out)
    nb = result.notebook_id
    assert nb != rich.notebook_id and result.name == "Solar" and not result.renamed
    assert logical_snapshot(conn, nb) == before
    assert result.report.ok and result.report.faiss == "5/5"
    assert [s for s, _ in result.stages] == [
        "Copying sources…",
        "Writing notebook data…",
        "Rebuilding indexes…",
        "Checking integrity…",
    ]
    # ids were remapped, relations kept
    new_sources = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM sources WHERE notebook_id = ? ORDER BY id", (nb,)
        )
    ]
    assert set(new_sources).isdisjoint({0}) and new_sources != rich.sources
    for sid in new_sources:
        assert settings.source_file(nb, sid).is_file()
        stored = conn.execute(
            "SELECT stored_path FROM sources WHERE id = ?", (sid,)
        ).fetchone()[0]
        assert (
            stored == f"notebooks/{nb}/sources/{sid}.pdf"
        )  # portable, forward slashes
    cit = conn.execute(
        "SELECT mc.chunk_id, mc.chunk_text, c.text, c.notebook_id "
        "FROM message_citations mc "
        "JOIN chunks c ON c.id = mc.chunk_id AND c.source_id = mc.source_id "
        "JOIN messages m ON m.id = mc.message_id JOIN conversations v "
        "ON v.id = m.conversation_id WHERE v.notebook_id = ?",
        (nb,),
    ).fetchone()
    assert cit[1] == cit[2] and cit[3] == nb  # [S1] resolves to the same passage
    ref = conn.execute(
        "SELECT r.node_key FROM message_memory_refs r "
        "JOIN messages m ON m.id = r.message_id "
        "JOIN conversations v ON v.id = m.conversation_id WHERE v.notebook_id = ?",
        (nb,),
    ).fetchone()[0]
    assert conn.execute(
        "SELECT 1 FROM memory_nodes WHERE notebook_id = ? AND key = ?", (nb, ref)
    ).fetchone()  # [M1] resolves in the rebuilt graph
    assert (
        conn.execute(
            "SELECT count(*) FROM memory_nodes WHERE notebook_id = ? "
            "AND node_type = 'correction'",
            (nb,),
        ).fetchone()[0]
        == 1
    )
    assert FaissStore(conn, settings).check(nb).ok
    # FTS finds imported text through the normal triggers
    for table, term in (
        ("chunks_fts", "Borealis"),
        ("notes_fts", "Reading"),
        ("annotations_fts", "fact"),
        ("messages_fts", "milliradians"),
    ):
        assert (
            conn.execute(
                f"SELECT count(*) FROM {table} WHERE {table} MATCH ?", (term,)
            ).fetchone()[0]
            >= 1
        ), table
    assert database_report(conn).ok


def test_name_collisions_and_explicit_names(conn, settings, tmp_path, rich):
    out = export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip").path
    names = [import_archive(conn, settings, out)[1][0].name for _ in range(2)]
    assert names == ["Solar (Imported)", "Solar (Imported 2)"]
    assert (
        import_archive(conn, settings, out, name="  My   copy ")[1][0].name == "My copy"
    )
    with pytest.raises(ArchiveError, match="already exists"):
        import_archive(conn, settings, out, name="Solar")
    assert unique_name(conn, "Solar") == "Solar (Imported 3)"
    create_notebook(conn, "x" * 100)
    long = unique_name(conn, "x" * 300)  # clipped to the 100-character limit
    assert long == "x" * 89 + " (Imported)" and len(long) == 100
    assert (
        conn.execute("SELECT count(*) FROM notebooks WHERE name = 'Solar'").fetchone()[
            0
        ]
        == 1
    )


def test_dangling_citations_never_attach_to_other_live_rows(
    conn, settings, tmp_path, rich
):
    store = FaissStore(conn, settings)
    delete_source(
        conn, settings, rich.notebook_id, rich.sources[0], store
    )  # cited source
    before = logical_snapshot(conn, rich.notebook_id)
    out = export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip").path
    other = create_notebook(conn, "Other")
    for i in range(3):  # live chunks with low ids that a naive import could hit
        ingest_pdf(
            conn,
            settings,
            other,
            make_pdf(tmp_path / f"o{i}.pdf", [f"Text {i}"]),
            FakeEmbedder(),
            store,
        )
    _, [result] = import_archive(conn, settings, out)
    after = logical_snapshot(conn, result.notebook_id)
    assert after["citations"] == before["citations"]
    assert after["citations"][0][-1] == 0  # still "source deleted", snapshot text kept
    row = conn.execute(
        "SELECT mc.chunk_id, mc.source_id FROM message_citations mc JOIN messages m "
        "ON m.id = mc.message_id JOIN conversations v ON v.id = m.conversation_id "
        "WHERE v.notebook_id = ?",
        (result.notebook_id,),
    ).fetchone()
    assert tuple(row) == (0, 0)


def test_reexport_is_logically_identical(conn, settings, tmp_path, rich):
    first = export_notebook(conn, settings, rich.notebook_id, tmp_path / "a.zip")
    _, [r] = import_archive(conn, settings, first.path, name="Copy")
    second = export_notebook(conn, settings, r.notebook_id, tmp_path / "b.zip")
    a, b = read_entries(first.path), read_entries(second.path)
    pa, pb = f"notebooks/{rich.notebook_id}", f"notebooks/{r.notebook_id}"
    for table in (
        "pages",
        "notes",
        "annotations",
        "annotation_rects",
        "memory_outcomes",
    ):

        def strip(raw: bytes) -> list[dict]:
            return [
                {k: v for k, v in json.loads(line).items() if not k.endswith("id")}
                for line in raw.splitlines()
            ]

        assert strip(a[f"{pa}/tables/{table}.jsonl"]) == strip(
            b[f"{pb}/tables/{table}.jsonl"]
        )
    pdfs = lambda e: sorted(v for k, v in e.items() if k.endswith(".pdf"))  # noqa: E731
    assert pdfs(a) == pdfs(b)
    assert first.counts == second.counts


def test_incompatible_or_invalid_embeddings_are_not_trusted(
    conn, settings, tmp_path, rich
):
    out = export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip").path
    entries = read_entries(out)
    nb = f"notebooks/{rich.notebook_id}"
    meta = json.loads(entries[f"{nb}/notebook.json"])
    meta["embed_model"] = "some/other-model"
    entries[f"{nb}/notebook.json"] = json.dumps(meta).encode()
    other_model = write_archive(tmp_path / "model.zip", entries)
    report = verify(other_model, settings)
    assert any("re-embedding" in w for w in report.warnings)
    _, [r] = import_archive(conn, settings, other_model)
    rows = conn.execute(
        "SELECT status, error FROM sources WHERE notebook_id = ?", (r.notebook_id,)
    ).fetchall()
    assert all(s == "failed" and "re-embed" in e for s, e in rows)
    assert (
        conn.execute(
            "SELECT count(*) FROM chunks WHERE notebook_id = ? AND embedding "
            "IS NOT NULL",
            (r.notebook_id,),
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT embed_model FROM notebooks WHERE id = ?", (r.notebook_id,)
        ).fetchone()[0]
        == EMBED_MODEL
    )
    assert r.report.faiss == "0/0" and r.reembed_sources == 2 and r.warnings

    entries = read_entries(out)
    import base64

    import numpy as np

    def poison(rows):
        rows[0]["embedding"] = base64.b64encode(
            np.full(384, np.nan, dtype="<f4").tobytes()
        ).decode()

    edit_rows(entries, f"{nb}/tables/chunks.jsonl", poison)
    bad = write_archive(tmp_path / "nan.zip", entries)
    assert any("invalid" in w for w in verify(bad, settings).warnings)
    _, [r2] = import_archive(conn, settings, bad)
    statuses = dict(
        conn.execute(
            "SELECT original_filename, status FROM sources WHERE notebook_id = ?",
            (r2.notebook_id,),
        ).fetchall()
    )
    assert sorted(statuses.values()) == ["failed", "ready"]  # only the poisoned source
    assert r2.report.faiss_ok


def test_imported_notebook_works_in_the_web_reader(tmp_path):
    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    conn = open_db(s.db_path)
    rich = build_rich_notebook(conn, s, tmp_path)
    out = export_notebook(conn, s, rich.notebook_id, tmp_path / "x.zip").path
    delete_notebook(conn, s, rich.notebook_id)
    _, [r] = import_archive(conn, s, out)
    sid = conn.execute(
        "SELECT min(id) FROM sources WHERE notebook_id = ?", (r.notebook_id,)
    ).fetchone()[0]
    conn.close()
    app = create_app(
        s,
        env={},
        llm_transport=httpx.MockTransport(
            lambda q: httpx.Response(200, json={"data": []})
        ),
    )
    with TestClient(app, base_url=BASE) as c:
        page = c.get(f"/notebooks/{r.notebook_id}/sources/{sid}/read")
        assert page.status_code == 200 and 'data-page="2"' in page.text  # reading state
        ann = c.get(
            f"/notebooks/{r.notebook_id}/sources/{sid}/annotations.json?page=2"
        ).json()
        assert ann["annotations"][0]["rects"] == [
            [0.1, 0.05, 0.4, 0.07],
            [0.1, 0.08, 0.3, 0.1],
        ]
        assert (
            c.get(
                f"/notebooks/{r.notebook_id}/sources/{sid}/pages/2/render"
            ).status_code
            == 200
        )
        chat = c.get(f"/notebooks/{r.notebook_id}/chat").text
        assert "0.1 milliradians" in chat and 'class="cite"' in chat
        assert (
            "Key &lt;i&gt;fact&lt;/i&gt;"
            in c.get(f"/notebooks/{r.notebook_id}/sources/{sid}/annotations").text
        )
        note = c.get(f"/notebooks/{r.notebook_id}/notes").text
        assert "Reading list" in note


# --- failures are all-or-nothing ----------------------------------------------------


@pytest.mark.parametrize("stage", ["Copying sources…", "database", "move", "index"])
def test_failures_leave_nothing_behind(conn, settings, tmp_path, rich, stage):
    out = export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip").path
    notebooks_before = conn.execute("SELECT count(*) FROM notebooks").fetchone()[0]
    files_before = notebook_files(settings)
    counts_before = {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in [*TABLES, "memory_nodes", "memory_edges"]
    }
    reader, report = open_and_verify(out, settings, conn)
    try:
        with pytest.raises(ImportFailedError, match="Nothing was imported"):
            import_notebook(conn, settings, reader, report.notebooks[0], _fail_at=stage)
    finally:
        reader.close()
    assert (
        conn.execute("SELECT count(*) FROM notebooks").fetchone()[0] == notebooks_before
    )
    assert notebook_files(settings) == files_before
    assert {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in counts_before
    } == counts_before
    assert not any(settings.archive_tmp_dir.glob("import-*"))
    assert database_report(conn).ok


def test_database_constraint_failure_rolls_back(conn, settings, tmp_path, rich):
    out = export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip").path
    entries = read_entries(out)
    nb = f"notebooks/{rich.notebook_id}"
    first = json.loads(entries[f"{nb}/tables/sources.jsonl"].splitlines()[0])

    def duplicate_sha(rows):  # UNIQUE(notebook_id, sha256) fails inside the transaction
        rows[1]["sha256"] = first["sha256"]

    edit_rows(entries, f"{nb}/tables/sources.jsonl", duplicate_sha)
    second_id = json.loads(entries[f"{nb}/tables/sources.jsonl"].splitlines()[1])["id"]
    entries[f"{nb}/sources/{second_id}.pdf"] = entries[
        f"{nb}/sources/{first['id']}.pdf"
    ]
    edit_rows(
        entries,
        f"{nb}/tables/sources.jsonl",
        lambda rows: rows[1].update(size_bytes=first["size_bytes"]),
    )
    bad = write_archive(tmp_path / "dup.zip", entries)
    before = conn.execute("SELECT count(*) FROM notebooks").fetchone()[0]
    files = notebook_files(settings)
    with pytest.raises(ImportFailedError, match="writing to the database"):
        import_archive(conn, settings, bad)
    assert conn.execute("SELECT count(*) FROM notebooks").fetchone()[0] == before
    assert notebook_files(settings) == files


def test_backup_import_is_all_or_nothing_across_notebooks(
    conn, settings, tmp_path, rich
):
    build_rich_notebook(conn, settings, tmp_path, name="Second")
    backup = create_backup(conn, settings, tmp_path / "b.zip").path
    entries = read_entries(backup)
    second = [n for n in entries if n.endswith("notebook.json")][1].split("/")[1]

    def bad_reference(
        rows,
    ):  # a soft reference is fine; break the graph rebuild instead
        rows[0]["content"] = rows[0]["content"]

    edit_rows(entries, f"notebooks/{second}/tables/notes.jsonl", bad_reference)
    good = write_archive(tmp_path / "g.zip", entries)
    before = conn.execute("SELECT count(*) FROM notebooks").fetchone()[0]
    reader, report = open_and_verify(good, settings, conn)
    reader.close()
    import litenotebook.archive.importer as importer

    calls = []
    real = importer.import_notebook

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            kwargs["_fail_at"] = "index"
        return real(*args, **kwargs)

    importer.import_notebook = flaky
    try:
        with pytest.raises(ImportFailedError):
            import_archive(conn, settings, good)
    finally:
        importer.import_notebook = real
    assert conn.execute("SELECT count(*) FROM notebooks").fetchone()[0] == before


# --- full backup, verify, restore ---------------------------------------------------


def test_backup_contains_every_notebook_and_verify_changes_nothing(
    conn, settings, tmp_path, rich
):
    build_rich_notebook(conn, settings, tmp_path, name="Second")
    conn.execute(
        "INSERT INTO settings (key, value) VALUES ('llm.model', 'qwen-test'), "
        "('llm.api_key', ?)",
        (SECRET,),
    )
    conn.commit()
    result = create_backup(conn, settings, tmp_path / "b.zip")
    assert result.counts["notebooks"] == 2 and result.counts["sources"] == 4
    db_before = settings.db_path.read_bytes()
    listing = sorted(str(p) for p in settings.data_dir.rglob("*"))
    report = verify(result.path, settings, conn)
    assert report.kind == "backup" and [n.meta["name"] for n in report.notebooks] == [
        "Solar",
        "Second",
    ]
    assert report.settings == [{"key": "llm.model", "value": "qwen-test"}]
    assert settings.db_path.read_bytes() == db_before
    assert sorted(str(p) for p in settings.data_dir.rglob("*")) == listing
    p = preview(report)
    assert p["sources"] == 4 and len(p["notebooks"]) == 2 and p["kind"] == "backup"


def test_newer_versions_are_refused_clearly(conn, settings, tmp_path, rich):
    out = export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip").path
    for field, value in (
        ("export_format_version", FORMAT_VERSION + 1),
        ("schema_version", 99),
    ):
        entries = read_entries(out)
        manifest = json.loads(entries["manifest.json"])
        manifest[field] = value
        entries["manifest.json"] = json.dumps(manifest).encode()
        newer = write_archive(tmp_path / f"{field}.zip", entries)
        with pytest.raises(UnsupportedArchiveError) as e:
            verify(newer, settings)
        assert str(e.value) == NEWER_VERSION_MESSAGE
        assert "newer LiteNotebook version" in str(e.value)


def test_restore_stages_then_switches_and_keeps_a_safety_copy(tmp_path):
    src = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "src")})
    conn = open_db(src.db_path)
    build_rich_notebook(conn, src, tmp_path, name="Alpha")
    build_rich_notebook(conn, src, tmp_path, name="Beta")
    conn.execute("INSERT INTO settings (key, value) VALUES ('rag.top_k', '5')")
    conn.commit()
    before = {
        n: logical_snapshot(conn, i)
        for i, n in conn.execute("SELECT id, name FROM notebooks").fetchall()
    }
    backup = create_backup(conn, src, tmp_path / "b.zip").path
    conn.close()

    live = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "live" / "data")})
    lc = open_db(live.db_path)
    create_notebook(lc, "Current work")
    lc.execute("INSERT INTO settings (key, value) VALUES ('llm.api_key', ?)", (SECRET,))
    lc.commit()
    lc.close()

    staged = stage_restore(live, backup)
    assert (
        staged.staging == staging_dir(live.data_dir)
        and (staged.staging / MARKER).is_file()
    )
    assert (
        pending_restore(live.data_dir)["archive_sha256"]
        == hashlib.sha256(backup.read_bytes()).hexdigest()
    )
    lc = open_db(live.db_path)  # nothing changed yet
    assert [r[0] for r in lc.execute("SELECT name FROM notebooks")] == ["Current work"]
    lc.close()
    with pytest.raises(ArchiveError, match="already staged"):
        stage_restore(live, backup)

    lock = InstanceLock(live.data_dir)
    assert lock.acquire()
    with pytest.raises(RestoreBusyError):
        apply_pending_restore(live.data_dir)  # a "server" holds the data directory
    lock.release()

    applied = apply_pending_restore(live.data_dir)
    assert applied is not None and applied.safety_dir.is_dir()
    assert (
        not staging_dir(live.data_dir).exists()
        and not (live.data_dir / MARKER).exists()
    )
    lc = open_db(live.db_path)
    after = {
        n: logical_snapshot(lc, i)
        for i, n in lc.execute("SELECT id, name FROM notebooks").fetchall()
    }
    assert after == before  # same names, same content
    settings_rows = dict(lc.execute("SELECT key, value FROM settings").fetchall())
    assert settings_rows["rag.top_k"] == "5" and settings_rows["llm.api_key"] == SECRET
    for nb_id in [r[0] for r in lc.execute("SELECT id FROM notebooks")]:
        assert notebook_report(lc, live, nb_id).ok
    lc.close()
    safety = open_db(applied.safety_dir / "litenotebook.sqlite3")
    assert [r[0] for r in safety.execute("SELECT name FROM notebooks")] == [
        "Current work"
    ]
    safety.close()
    assert apply_pending_restore(live.data_dir) is None


def test_corrupt_backup_is_rejected_without_touching_data(
    conn, settings, tmp_path, rich
):
    backup = create_backup(conn, settings, tmp_path / "b.zip").path
    entries = read_entries(backup)
    pdf = next(n for n in entries if n.endswith(".pdf"))
    entries[pdf] = entries[pdf][:-10] + b"0123456789"
    corrupt = write_archive(tmp_path / "c.zip", entries, fix_checksums=False)
    with pytest.raises(ArchiveError, match="Checksum mismatch"):
        stage_restore(settings, corrupt)
    assert not staging_dir(settings.data_dir).exists()
    assert conn.execute("SELECT count(*) FROM notebooks").fetchone()[0] == 1
    notebook_export = export_notebook(
        conn, settings, rich.notebook_id, tmp_path / "n.zip"
    )
    with pytest.raises(ArchiveError, match="single-notebook export"):
        stage_restore(settings, notebook_export.path)


def test_incomplete_staging_is_discarded_and_cancel_works(settings, tmp_path):
    staging = staging_dir(settings.data_dir)
    (staging / "notebooks").mkdir(parents=True)
    assert apply_pending_restore(settings.data_dir) is None and not staging.exists()
    staging.mkdir()
    (staging / MARKER).write_text("{}")
    assert cancel_restore(settings.data_dir) and not staging.exists()
    assert not cancel_restore(settings.data_dir)


def test_create_app_applies_a_staged_restore_at_startup(tmp_path):
    src = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "src")})
    conn = open_db(src.db_path)
    build_rich_notebook(conn, src, tmp_path, name="Restored")
    backup = create_backup(conn, src, tmp_path / "b.zip").path
    conn.close()
    live = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "live")})
    open_db(live.db_path).close()
    stage_restore(live, backup)
    app = create_app(
        live,
        env={},
        llm_transport=httpx.MockTransport(
            lambda q: httpx.Response(200, json={"data": []})
        ),
    )
    with TestClient(app, base_url=BASE) as c:
        assert "Restored" in c.get("/").text
    assert list(tmp_path.glob("live.pre-restore-*"))


# --- web ----------------------------------------------------------------------------


@pytest.fixture
def web(tmp_path):
    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    conn = open_db(s.db_path)
    rich = build_rich_notebook(conn, s, tmp_path)
    conn.close()
    calls = []

    def factory():
        calls.append(1)
        return FakeEmbedder()

    def llama(request):
        assert request.url.path.endswith("/models"), "no generation in Phase 10"
        return httpx.Response(200, json={"data": []})

    app = create_app(
        s, env={}, embedder_factory=factory, llm_transport=httpx.MockTransport(llama)
    )
    with TestClient(app, base_url=BASE) as c:
        yield {"c": c, "s": s, "nb": rich.notebook_id, "calls": calls, "tmp": tmp_path}


def upload(c, url, path, **fields):
    return c.post(
        url,
        headers=ORIGIN,
        data=fields,
        files={"file": (path.name, path.read_bytes(), "application/zip")},
    )


def test_web_export_import_verify(web):
    c, nb, tmp = web["c"], web["nb"], web["tmp"]
    page = c.get(f"/notebooks/{nb}/export").text
    assert "Export notebook" in page and "Sources</dt><dd>2 PDFs" in page
    assert str(web["s"].data_dir) not in page
    r = c.post(f"/notebooks/{nb}/export", headers=ORIGIN)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert re.search(
        r'filename="litenotebook-Solar-\d{8}-\d{4}\.zip"',
        r.headers["content-disposition"],
    )
    archive = tmp / "dl.zip"
    archive.write_bytes(r.content)
    assert not any(web["s"].archive_tmp_dir.glob("*.zip"))  # removed after sending
    v = upload(c, "/backup/verify", archive)
    assert (
        v.status_code == 200
        and "The archive is valid" in v.text
        and "Nothing was" in v.text
    )
    imported = upload(c, "/backup/import", archive)
    assert imported.status_code == 200 and "Solar (Imported)" in imported.text
    assert "A notebook named “Solar” already existed" in imported.text
    assert "5/5 ✓ rebuilt" in imported.text and "Checking integrity…" in imported.text
    named = upload(c, "/backup/import", archive, name="Field copy")
    assert "Field copy" in named.text
    bad = tmp / "bad.zip"
    bad.write_bytes(b"PK\x03\x04 not really a zip")
    r = upload(c, "/backup/import", bad)
    assert r.status_code == 422 and "not a valid ZIP" in r.text
    assert not any(web["s"].archive_tmp_dir.iterdir())  # uploads cleaned up
    b = c.post("/backup/create", headers=ORIGIN)
    assert b.status_code == 200 and b.content[:2] == b"PK"
    assert web["calls"] == []  # MiniLM never started


def test_web_restore_requires_strong_confirmation(web):
    c, tmp, s = web["c"], web["tmp"], web["s"]
    backup = tmp / "b.zip"
    backup.write_bytes(c.post("/backup/create", headers=ORIGIN).content)
    p = upload(c, "/backup/restore/preview", backup)
    assert p.status_code == 200 and "Replace all current data?" in p.text
    token = re.search(r'name="token" value="([0-9a-f]{32})"', p.text).group(1)
    wrong = c.post(
        "/backup/restore/confirm",
        headers=ORIGIN,
        data={"token": token, "confirm": "restore", "understand": "1"},
    )
    assert wrong.status_code == 422 and "not confirmed" in wrong.text
    assert pending_restore(s.data_dir) is None
    p = upload(c, "/backup/restore/preview", backup)
    token = re.search(r'name="token" value="([0-9a-f]{32})"', p.text).group(1)
    ok = c.post(
        "/backup/restore/confirm",
        headers=ORIGIN,
        data={"token": token, "confirm": "RESTORE", "understand": "1"},
    )
    assert ok.status_code == 200 and "Restart LiteNotebook to finish" in ok.text
    assert pending_restore(s.data_dir) is not None
    assert "A restore is waiting" in c.get("/backup").text
    assert apply_pending_restore  # the running server holds the lock:
    with pytest.raises(RestoreBusyError):
        apply_pending_restore(s.data_dir)
    r = c.post("/backup/restore/cancel", headers=ORIGIN, follow_redirects=False)
    assert r.status_code == 303 and pending_restore(s.data_dir) is None
    replay = c.post(
        "/backup/restore/confirm",
        headers=ORIGIN,
        data={"token": token, "confirm": "RESTORE", "understand": "1"},
    )
    assert replay.status_code == 422 and "expired" in replay.text
    single = tmp / "n.zip"
    single.write_bytes(c.post(f"/notebooks/{web['nb']}/export", headers=ORIGIN).content)
    assert "single-notebook export" in upload(c, "/backup/restore/preview", single).text


def test_backup_page_and_links(web):
    c, nb = web["c"], web["nb"]
    page = c.get("/backup").text
    for text in (
        "Export a notebook",
        "Import a notebook",
        "Create a full backup",
        "Verify a backup",
        "Restore from a full backup",
        'class="panel danger-zone"',
    ):
        assert text in page
    assert 'href="/backup"' in c.get("/settings").text
    assert f'href="/notebooks/{nb}/export"' in c.get(f"/notebooks/{nb}").text
    labels = [x["label"] for x in c.get(f"/palette.json?nb={nb}").json()["commands"]]
    assert "Export this notebook" in labels and "Backup & Restore" in labels


# --- performance & lazy loading -----------------------------------------------------


def test_large_pdfs_stream_in_bounded_memory(conn, settings, tmp_path):
    nb = create_notebook(conn, "Big")
    big = 48 * 1024 * 1024
    path = tmp_path / "big.pdf"
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4\n")
        chunk = bytes(range(256)) * 4096
        while f.tell() < big:
            f.write(chunk)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cur = conn.execute(
        "INSERT INTO sources (notebook_id, title, original_filename, stored_path, "
        "mime, sha256, size_bytes, page_count, status, stage) "
        "VALUES (?, 'big', 'big.pdf', '', 'application/pdf', ?, ?, 1, "
        "'ready', 'ready')",
        (nb.id, sha, path.stat().st_size),
    )
    dest = settings.source_file(nb.id, cur.lastrowid)
    dest.parent.mkdir(parents=True)
    path.rename(dest)
    conn.commit()
    tracemalloc.start()
    out = export_notebook(conn, settings, nb.id, tmp_path / "big.zip").path
    _, export_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    _, [r] = import_archive(conn, settings, out, name="Big copy")
    _, import_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert export_peak < 8 * 1024 * 1024, export_peak
    assert import_peak < 16 * 1024 * 1024, import_peak
    assert (
        settings.source_file(
            r.notebook_id,
            conn.execute(
                "SELECT id FROM sources WHERE notebook_id = ?", (r.notebook_id,)
            ).fetchone()[0],
        )
        .stat()
        .st_size
        == dest.stat().st_size
    )


def test_archive_work_never_starts_minilm_or_imports_torch(tmp_path):
    code = f"""
import sys, httpx
sys.path.insert(0, {str(Path(__file__).parent)!r})
from pathlib import Path
from fastapi.testclient import TestClient
from archive_fixture import build_rich_notebook
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.archive.export import export_notebook, create_backup
from litenotebook.archive.importer import import_archive
from litenotebook.archive.reader import verify_archive
from litenotebook.archive.format import Limits
from litenotebook.web.app import create_app
tmp = Path({str(tmp_path)!r})
s = load_settings({{"LITENOTEBOOK_DATA_DIR": str(tmp / "d")}})
c = open_db(s.db_path)
build_rich_notebook(c, s, tmp)
for m in ("torch", "sentence_transformers", "transformers"):
    assert m not in sys.modules, m
out = export_notebook(c, s, 1, tmp / "x.zip").path
verify_archive(out, Limits.from_settings(s), current_schema={CURRENT_SCHEMA})
import_archive(c, s, out)
create_backup(c, s, tmp / "b.zip")
c.close()
calls = []
def factory():
    calls.append(1)
    raise AssertionError("must not create the embedder")
def llama(r):
    assert r.url.path.endswith("/models")
    return httpx.Response(200, json={{"data": []}})
app = create_app(s, env={{}}, embedder_factory=factory,
                 llm_transport=httpx.MockTransport(llama))
with TestClient(app, base_url="{BASE}") as cl:
    for u in ("/backup", "/notebooks/1/export", "/settings"):
        assert cl.get(u).status_code == 200, u
    data = cl.post("/notebooks/1/export", headers={{"Origin": "{BASE}"}}).content
    (tmp / "w.zip").write_bytes(data)
    r = cl.post("/backup/import", headers={{"Origin": "{BASE}"}},
                files={{"file": ("w.zip", data, "application/zip")}})
    assert r.status_code == 200, r.text[:300]
heavy = ("torch", "sentence_transformers", "transformers")
print("HEAVY", [m for m in heavy if m in sys.modules], len(calls))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr[-2500:]
    assert "HEAVY [] 0" in out.stdout


def test_remap_key_rules():
    maps = {
        "messages": {5: 50},
        "notes": {2: 20},
        "message_citations": {1: 10},
        "note_links": {3: 30},
        "conversations": {7: 70},
    }
    assert remap_key("question:5", maps) == "question:50"
    assert remap_key("answer:5", maps) == "answer:50"
    assert remap_key("note:2", maps) == "note:20"
    assert remap_key("note_evidence:3", maps) == "note_evidence:30"
    assert remap_key("evidence:1", maps) == "evidence:10"
    assert remap_key("conversation:7", maps) == "conversation:70"
    assert remap_key("concept:heliostat", maps) == "concept:heliostat"
    assert (
        remap_key("answer:999", maps) == "answer:gone-999"
    )  # never matches a live node
    sqlite3.connect(":memory:").close()
