"""Backup / import / restore disaster cases (Phase 16).

Complements ``test_archive.py`` and ``test_archive_security.py`` (hostile
entry names, bombs, limits, row validation, injected stage failures). Here:
corrupted members of a real format-3 archive, consistent tampering, a full
disk at each write, an export destination that vanishes, interrupted import
and restore, safety-folder name collisions, and a full current-format backup
restored into an isolated clean data folder.

Every failure must leave either no partial notebook or a documented,
recoverable state.
"""

from __future__ import annotations

import base64
import errno
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from archive_fixture import (
    build_rich_notebook,
    logical_snapshot,
    read_entries,
    write_archive,
)
from litenotebook.archive import export as export_mod
from litenotebook.archive import restore as restore_mod
from litenotebook.archive.export import create_backup, export_notebook
from litenotebook.archive.format import TABLES, ArchiveError, UnsupportedArchiveError
from litenotebook.archive.importer import ImportFailedError, import_archive
from litenotebook.archive.integrity import database_report, notebook_report
from litenotebook.archive.reader import ArchiveReader
from litenotebook.archive.restore import (
    MARKER,
    apply_pending_restore,
    stage_restore,
    staging_dir,
)
from litenotebook.config import load_settings
from litenotebook.db import open_db

FIXTURES = Path(__file__).parent / "fixtures"
V3 = FIXTURES / "phase14_notebook_v3.zip"


def _state(conn: sqlite3.Connection, settings) -> dict[str, object]:  # type: ignore[no-untyped-def]
    counts = {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in [*TABLES, "memory_nodes", "memory_edges"]
    }
    root = settings.notebooks_dir
    files = (
        sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
        if root.exists()
        else []
    )
    return {"counts": counts, "files": files}


def _nothing_changed(conn, settings, before) -> None:  # type: ignore[no-untyped-def]
    assert _state(conn, settings) == before
    assert (
        not list(settings.archive_tmp_dir.glob("import-*"))
        if settings.archive_tmp_dir.exists()
        else True
    )
    assert database_report(conn).ok


def _enospc() -> OSError:
    return OSError(errno.ENOSPC, "No space left on device")


# --- corrupted or tampered archive members -----------------------------------------


def _corrupt(entries: dict[str, bytes], what: str) -> None:
    if what == "manifest":
        entries["manifest.json"] = b'{"format": "litenotebook-archive", "export_form'
    elif what == "pdf":
        name = next(n for n in entries if n.endswith(".pdf"))
        entries[name] = entries[name][:-20] + b"X" * 20
    elif what == "audio":
        name = next(n for n in entries if n.endswith(".wav"))
        entries[name] = entries[name][:44] + bytes(len(entries[name]) - 44)
    elif what == "embedding":
        name = next(n for n in entries if n.endswith("chunks.jsonl"))
        rows = [json.loads(x) for x in entries[name].decode().splitlines() if x]
        blob = bytearray(base64.b64decode(rows[0]["embedding"]))
        blob[:8] = b"\xff" * 8
        rows[0]["embedding"] = base64.b64encode(bytes(blob)).decode()
        entries[name] = "".join(json.dumps(r) + "\n" for r in rows).encode()
    else:
        raise AssertionError(what)


@pytest.mark.parametrize("what", ["manifest", "pdf", "audio", "embedding"])
def test_corrupted_members_import_nothing(conn, settings, tmp_path, what):
    entries = read_entries(V3)
    _corrupt(entries, what)
    bad = write_archive(tmp_path / f"{what}.zip", entries, fix_checksums=False)
    before = _state(conn, settings)
    with pytest.raises(ArchiveError):
        import_archive(conn, settings, bad)
    _nothing_changed(conn, settings, before)


@pytest.mark.parametrize("what", ["pdf", "audio"])
def test_consistently_tampered_files_import_nothing(conn, settings, tmp_path, what):
    """Checksums recomputed by an attacker: the per-row SHA-256 still catches it."""
    entries = read_entries(V3)
    _corrupt(entries, what)
    bad = write_archive(tmp_path / f"{what}.zip", entries, fix_checksums=True)
    before = _state(conn, settings)
    with pytest.raises(ArchiveError):
        import_archive(conn, settings, bad)
    _nothing_changed(conn, settings, before)


def test_tampered_embedding_is_not_trusted_but_import_is_safe(conn, settings, tmp_path):
    entries = read_entries(V3)
    _corrupt(entries, "embedding")
    bad = write_archive(tmp_path / "emb.zip", entries, fix_checksums=True)
    try:
        _, results = import_archive(conn, settings, bad)
    except ArchiveError:
        return  # rejected outright: also acceptable
    nb = results[0].notebook_id
    assert notebook_report(conn, settings, nb).ok
    # a non-normalised vector is never stored as a trusted embedding
    rows = conn.execute(
        "SELECT embedding FROM chunks WHERE notebook_id = ?", (nb,)
    ).fetchall()
    assert any(r[0] is None for r in rows)


def test_truncated_and_future_archives(conn, settings, tmp_path):
    raw = V3.read_bytes()
    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(raw[: len(raw) // 2])
    before = _state(conn, settings)
    with pytest.raises(ArchiveError):
        import_archive(conn, settings, truncated)
    entries = read_entries(V3)
    manifest = json.loads(entries["manifest.json"])
    manifest["export_format_version"] = 99
    entries["manifest.json"] = json.dumps(manifest).encode()
    future = write_archive(tmp_path / "future.zip", entries, fix_checksums=True)
    with pytest.raises(UnsupportedArchiveError):
        import_archive(conn, settings, future)
    _nothing_changed(conn, settings, before)


# --- full disk, vanishing destination, interruption ---------------------------------


def _full_disk_on(monkeypatch, suffix: str) -> None:  # type: ignore[no-untyped-def]
    """Writes of staged files ending in ``suffix`` fail with ENOSPC."""
    from litenotebook.archive import importer

    real = importer.create_exclusive

    class Full:
        def __init__(self, f):  # type: ignore[no-untyped-def]
            self.f = f

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *exc):  # type: ignore[no-untyped-def]
            self.f.close()
            return False

        def write(self, data):  # type: ignore[no-untyped-def]
            self.f.write(data[: len(data) // 2])  # a half-written file
            raise _enospc()

    def create(path, *a, **k):  # type: ignore[no-untyped-def]
        f = real(path, *a, **k)
        return Full(f) if str(path).endswith(suffix) else f

    monkeypatch.setattr(importer, "create_exclusive", create)


def test_disk_full_while_copying_import_files(conn, settings, tmp_path, monkeypatch):
    _full_disk_on(monkeypatch, ".wav")
    before = _state(conn, settings)
    with pytest.raises(ImportFailedError, match="Nothing was imported"):
        import_archive(conn, settings, V3)
    _nothing_changed(conn, settings, before)


def test_interrupted_import_leaves_nothing(conn, settings, tmp_path, monkeypatch):
    real_stream = ArchiveReader.stream

    def stream(self, name, *args, **kw):  # type: ignore[no-untyped-def]
        yield from real_stream(self, name, *args, **kw)
        if name.endswith(".pdf"):
            raise KeyboardInterrupt  # the user pressed Ctrl+C / the app is closing

    monkeypatch.setattr(ArchiveReader, "stream", stream)
    before = _state(conn, settings)
    with pytest.raises(KeyboardInterrupt):
        import_archive(conn, settings, V3)
    _nothing_changed(conn, settings, before)


def test_disk_full_during_export_leaves_no_file(conn, settings, tmp_path, monkeypatch):
    rich = build_rich_notebook(conn, settings, tmp_path)
    out = tmp_path / "exports" / "nb.zip"
    real = export_mod._write_notebook

    def full(w, *a, **k):  # type: ignore[no-untyped-def]
        real(w, *a, **k)
        raise _enospc()

    monkeypatch.setattr(export_mod, "_write_notebook", full)
    with pytest.raises(OSError, match="No space"):
        export_notebook(conn, settings, rich.notebook_id, out)
    assert list(out.parent.iterdir()) == []  # no partial, no "valid-looking" zip


def test_export_destination_disappears(conn, settings, tmp_path, monkeypatch):
    rich = build_rich_notebook(conn, settings, tmp_path)
    out = tmp_path / "usb stick" / "nb.zip"
    real = export_mod._write_notebook

    def unplugged(w, *a, **k):  # type: ignore[no-untyped-def]
        result = real(w, *a, **k)
        shutil.rmtree(out.parent, ignore_errors=True)  # the drive was removed
        return result

    monkeypatch.setattr(export_mod, "_write_notebook", unplugged)
    before = _state(conn, settings)
    with pytest.raises(OSError):
        export_notebook(conn, settings, rich.notebook_id, out)
    assert not out.exists()
    assert _state(conn, settings) == before  # exporting never changes data


def test_disk_full_while_staging_restore(conn, settings, tmp_path, monkeypatch):
    rich = build_rich_notebook(conn, settings, tmp_path)
    backup = create_backup(conn, settings, tmp_path / "b.zip").path
    before = logical_snapshot(conn, rich.notebook_id)
    _full_disk_on(monkeypatch, ".pdf")
    with pytest.raises(ImportFailedError):
        stage_restore(settings, backup)
    assert not staging_dir(settings.data_dir).exists()
    assert apply_pending_restore(settings.data_dir) is None
    assert logical_snapshot(conn, rich.notebook_id) == before


def _two_installs(tmp_path: Path):  # type: ignore[no-untyped-def]
    src = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "src")})
    conn = open_db(src.db_path)
    build_rich_notebook(conn, src, tmp_path, name="Alpha")
    backup = create_backup(conn, src, tmp_path / "b.zip").path
    conn.close()
    live = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "live" / "data")})
    lc = open_db(live.db_path)
    lc.execute(
        "INSERT INTO notebooks (name, embed_model, embed_dim) "
        "VALUES ('Current', 'm', 384)"
    )
    lc.commit()
    lc.close()
    return live, backup


def _names(settings) -> list[str]:  # type: ignore[no-untyped-def]
    c = open_db(settings.db_path)
    try:
        return [r[0] for r in c.execute("SELECT name FROM notebooks ORDER BY name")]
    finally:
        c.close()


def test_restore_interrupted_between_the_two_renames_finishes(tmp_path):
    """Crash after <data> was moved aside but before staging was moved in."""
    live, backup = _two_installs(tmp_path)
    stage_restore(live, backup)
    aside = live.data_dir.with_name("data.pre-restore-20260101T000000Z")
    live.data_dir.rename(aside)  # the first rename happened, then power loss
    applied = apply_pending_restore(live.data_dir)
    assert applied is not None and applied.safety_dir is None
    assert _names(live) == ["Alpha"]
    assert aside.is_dir()  # the previous data is still there to recover


def test_restore_interrupted_at_the_very_end_is_clean(tmp_path):
    live, backup = _two_installs(tmp_path)
    stage_restore(live, backup)
    applied = apply_pending_restore(live.data_dir)
    assert applied is not None
    (live.data_dir / MARKER).write_text("{}")  # left behind by a crash at the end
    assert apply_pending_restore(live.data_dir) is None
    assert not (live.data_dir / MARKER).exists()
    assert _names(live) == ["Alpha"]


def test_existing_safety_folder_is_never_reused(tmp_path, monkeypatch):
    live, backup = _two_installs(tmp_path)
    fixed = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def,override]
            return fixed

    monkeypatch.setattr(restore_mod, "datetime", Frozen)
    taken = live.data_dir.with_name("data.pre-restore-20261010T120000Z")
    taken.mkdir()
    (taken / "keep.txt").write_text("earlier safety copy")
    empty = live.data_dir.with_name("data.pre-restore-20261010T120000Z-2")
    empty.mkdir()  # even an empty folder is not replaced
    stage_restore(live, backup)
    applied = apply_pending_restore(live.data_dir)
    assert applied is not None
    assert applied.safety_dir.name == "data.pre-restore-20261010T120000Z-3"
    assert (taken / "keep.txt").read_text() == "earlier safety copy"
    assert list(empty.iterdir()) == []
    assert _names(live) == ["Alpha"]
    c = open_db(applied.safety_dir / "litenotebook.sqlite3")
    assert [r[0] for r in c.execute("SELECT name FROM notebooks")] == ["Current"]
    c.close()


def test_full_current_format_backup_restores_into_a_clean_folder(tmp_path):
    """Everything (PDFs, chat, notes, annotations, quiz-free audio fixture)
    round-trips into a brand-new data folder that never existed."""
    src = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "src")})
    conn = open_db(src.db_path)
    build_rich_notebook(conn, src, tmp_path, name="Alpha")
    import_archive(conn, src, V3)
    before = {
        n: logical_snapshot(conn, i)
        for i, n in conn.execute("SELECT id, name FROM notebooks").fetchall()
    }
    audio_before = conn.execute("SELECT sha256 FROM audio_assets").fetchall()
    backup = create_backup(conn, src, tmp_path / "full.zip").path
    conn.close()

    clean = load_settings(
        {"LITENOTEBOOK_DATA_DIR": str(tmp_path / "elsewhere" / "fresh")}
    )
    assert not clean.data_dir.exists()
    stage_restore(clean, backup)
    applied = apply_pending_restore(clean.data_dir)
    assert applied is not None and applied.safety_dir is None
    c = open_db(clean.db_path)
    try:
        after = {
            n: logical_snapshot(c, i)
            for i, n in c.execute("SELECT id, name FROM notebooks").fetchall()
        }
        assert after == before
        assert c.execute("SELECT sha256 FROM audio_assets").fetchall() == audio_before
        for (nb,) in c.execute("SELECT id FROM notebooks").fetchall():
            assert notebook_report(c, clean, nb).ok
        assert database_report(c).ok
    finally:
        c.close()
    wavs = list(clean.notebooks_dir.rglob("*.wav"))
    assert len(wavs) == len(audio_before) and all(w.stat().st_size > 44 for w in wavs)
