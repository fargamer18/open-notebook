"""Upgrade paths from every earlier schema to the current one.

Two kinds of starting databases:

* real databases written by the Phase 12 (schema 8) and Phase 13 (schema 9)
  code (``tests/fixtures/db_schema*.sqlite3.gz``);
* databases at every earlier schema (1–9) built from the migration files
  themselves, with data inserted at the schema where each table first
  appeared, so the data travels through every later migration.

Each upgrade checks the safety copy, ``integrity_check``,
``foreign_key_check``, the FTS index, row counts and that the schema (tables,
indexes, triggers) equals a fresh database. A failing migration must leave the
database at its last good version with its data and the safety copy intact.
"""

from __future__ import annotations

import gzip
import shutil
import sqlite3
import struct
from pathlib import Path

import pytest

from litenotebook import db as dbmod
from litenotebook.db import MigrationError, available_migrations, connect, migrate

FIXTURES = Path(__file__).parent / "fixtures"
LATEST = len(available_migrations())


def _apply(conn: sqlite3.Connection, upto: int, start: int = 0) -> None:
    """Apply migrations the way ``db.migrate`` does (foreign keys off)."""
    conn.execute("PRAGMA foreign_keys = OFF")
    for version, path in available_migrations()[start:upto]:
        conn.executescript(
            f"BEGIN;\n{path.read_text('utf-8')}\nPRAGMA user_version = {version};"
            "\nCOMMIT;"
        )
    conn.execute("PRAGMA foreign_keys = ON")


def _vec(seed: int) -> bytes:
    values = [((seed * 31 + i) % 97) / 97.0 for i in range(384)]
    norm = sum(v * v for v in values) ** 0.5
    return struct.pack("<384f", *(v / norm for v in values))


def build_db(path: Path, version: int) -> dict[str, int]:
    """A database at ``version`` with representative data; returns counts."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    _apply(conn, 1)
    for nb in (1, 2):
        conn.execute(
            "INSERT INTO notebooks (id, name, embed_model, embed_dim) "
            "VALUES (?, ?, ?, 384)",
            (nb, f"Notebook {nb}", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        conn.execute(
            "INSERT INTO sources (id, notebook_id, title, original_filename, "
            "stored_path, mime, sha256, size_bytes, page_count, status) "
            "VALUES (?, ?, ?, ?, ?, 'application/pdf', ?, 100, 2, 'ready')",
            (
                nb,
                nb,
                f"Atlas {nb}",
                f"atlas {nb}.pdf",
                f"notebooks/{nb}/sources/{nb}.pdf",
                f"{nb:064x}",
            ),
        )
        for page in (1, 2):
            text = f"Velt river page {page} notebook {nb}. Orrin mountains glacier."
            cur = conn.execute(
                "INSERT INTO pages (source_id, page_no, text, method) "
                "VALUES (?, ?, ?, 'text')",
                (nb, page, text),
            )
            conn.execute(
                "INSERT INTO chunks (notebook_id, source_id, page_id, page_no, ord, "
                "char_start, char_end, text, n_tokens, embedding) "
                "VALUES (?, ?, ?, ?, ?, 0, ?, ?, 12, ?)",
                (
                    nb,
                    nb,
                    cur.lastrowid,
                    page,
                    page - 1,
                    len(text),
                    text,
                    _vec(nb * 10 + page),
                ),
            )
    conn.commit()
    counts = {"notebooks": 2, "sources": 2, "pages": 4, "chunks": 4}
    if version >= 4:
        _apply(conn, 4, 1)
        conn.execute(
            "INSERT INTO conversations (id, notebook_id, title) VALUES (1, 1, 'Rivers')"
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, status) "
            "VALUES (1, 1, 'user', 'Where does the Velt rise?', 'complete')"
        )
        conn.execute(
            "INSERT INTO messages "
            "(id, conversation_id, role, content, status, reply_to) "
            "VALUES (2, 1, 'assistant', 'In the Orrin mountains [S1].', 'complete', 1)"
        )
        conn.execute(
            "INSERT INTO message_citations (message_id, label, ordinal, chunk_id, "
            "source_id, source_filename, page_no, chunk_ord, chunk_text) "
            "VALUES (2, 1, 0, 1, 1, 'atlas 1.pdf', 1, 0, 'Velt river page 1')"
        )
        conn.commit()
        counts.update(conversations=1, messages=2, message_citations=1)
    if version >= 7:
        _apply(conn, 7, 4)
        conn.execute(
            "INSERT INTO notes (id, notebook_id, title, content) "
            "VALUES (1, 1, 'Harbour', 'Quillport trams are violet.')"
        )
        conn.commit()
        counts["notes"] = 1
    _apply(conn, version, max(1, 4 if version >= 4 else 1, 7 if version >= 7 else 1))
    assert conn.execute("PRAGMA user_version").fetchone()[0] == version
    conn.close()
    return counts


def _schema(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    return {
        (row[0], row[1])
        for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        )
    }


@pytest.fixture(scope="module")
def fresh_schema(tmp_path_factory: pytest.TempPathFactory) -> set[tuple[str, str]]:
    conn = connect(tmp_path_factory.mktemp("fresh") / "fresh.sqlite3")
    migrate(conn)
    try:
        return _schema(conn)
    finally:
        conn.close()


def _check_upgraded(
    path: Path,
    start: int,
    counts: dict[str, int],
    fresh: set[tuple[str, str]],
    fts_word: str,
) -> None:
    conn = connect(path)
    try:
        assert migrate(conn) == LATEST
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('integrity-check')")
        hits = conn.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (fts_word,)
        ).fetchone()[0]
        assert hits >= 1
        for table, n in counts.items():
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == n, (
                table
            )
        assert _schema(conn) == fresh  # every table, index and trigger present
    finally:
        conn.close()
    backups = sorted((path.parent / "backups").glob("*.sqlite3"))
    assert [b.name.split("-from-")[1].split("-")[0] for b in backups] == [str(start)]
    assert backups[0].name.startswith(f"before-schema-{LATEST}-from-{start}-")
    copy = sqlite3.connect(backups[0])
    try:
        assert copy.execute("PRAGMA user_version").fetchone()[0] == start
        assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            copy.execute("SELECT count(*) FROM chunks").fetchone()[0]
            == counts["chunks"]
        )
    finally:
        copy.close()
    assert not list((path.parent / "backups").glob(".*partial"))


@pytest.mark.parametrize("start", range(1, LATEST))
def test_upgrade_from_every_schema(
    tmp_path: Path, start: int, fresh_schema: set[tuple[str, str]]
) -> None:
    path = tmp_path / "litenotebook.sqlite3"
    counts = build_db(path, start)
    _check_upgraded(path, start, counts, fresh_schema, "glacier")


@pytest.mark.parametrize(
    ("fixture", "start"),
    [("db_schema8_phase12.sqlite3.gz", 8), ("db_schema9_phase13.sqlite3.gz", 9)],
)
def test_upgrade_real_databases_from_older_releases(
    tmp_path: Path, fixture: str, start: int, fresh_schema: set[tuple[str, str]]
) -> None:
    path = tmp_path / "litenotebook.sqlite3"
    with gzip.open(FIXTURES / fixture) as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    counts = {"notebooks": 1, "sources": 1, "chunks": 1, "notes": 1}
    _check_upgraded(path, start, counts, fresh_schema, "Velt")


def _broken_migrations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_version: int
) -> None:
    """Real migrations up to ``bad_version - 1``, then one that fails halfway."""
    real = available_migrations()[: bad_version - 1]
    bad = tmp_path / f"{bad_version:04d}_broken.sql"
    bad.write_text(
        "CREATE TABLE half_done (id INTEGER);\n"
        "DELETE FROM chunks;\n"  # destructive step that must be rolled back
        "INSERT INTO no_such_table VALUES (1);\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dbmod, "available_migrations", lambda: [*real, (bad_version, bad)]
    )


def test_failed_migration_rolls_back_and_keeps_the_safety_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "litenotebook.sqlite3"
    counts = build_db(path, 8)
    # 8 → 9 (real) succeeds, 9 → 10 (broken) fails halfway
    _broken_migrations(monkeypatch, tmp_path, 10)
    conn = connect(path)
    try:
        with pytest.raises(MigrationError, match="0010_broken"):
            migrate(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 9
        assert (
            conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            == counts["chunks"]
        )
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'half_done'"
        ).fetchone()
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
    backups = list((tmp_path / "backups").glob("before-schema-10-from-8-*.sqlite3"))
    assert len(backups) == 1
    monkeypatch.undo()  # the fixed release upgrades the remaining step
    conn = connect(path)
    try:
        assert migrate(conn) == LATEST
        assert (
            conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            == counts["chunks"]
        )
    finally:
        conn.close()
    assert len(list((tmp_path / "backups").glob("before-schema-*.sqlite3"))) == 2


def test_safety_copy_failure_stops_the_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No copy (e.g. disk full) → no migration; the database is untouched."""
    path = tmp_path / "litenotebook.sqlite3"
    counts = build_db(path, 8)

    def disk_full(c: sqlite3.Connection, current: int, target: int) -> Path | None:
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(dbmod, "backup_before_migration", disk_full)
    conn = connect(path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            migrate(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 8
        assert (
            conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            == counts["chunks"]
        )
    finally:
        conn.close()


def test_partial_safety_copy_is_removed(tmp_path: Path) -> None:
    path = tmp_path / "litenotebook.sqlite3"
    build_db(path, 8)
    conn = connect(path)

    real_connect = sqlite3.connect
    try:
        with pytest.MonkeyPatch.context() as mp:

            def connect_then_fail(
                target: object, *a: object, **k: object
            ) -> sqlite3.Connection:
                dest = real_connect(target, *a, **k)  # type: ignore[arg-type]
                Path(str(target)).write_bytes(b"half a database")
                dest.close()
                raise sqlite3.OperationalError("database or disk is full")

            mp.setattr(dbmod.sqlite3, "connect", connect_then_fail)
            with pytest.raises(sqlite3.OperationalError):
                dbmod.backup_before_migration(conn, 8, LATEST)
        assert list((tmp_path / "backups").iterdir()) == []
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 8
    finally:
        conn.close()


def test_safety_copies_never_overwrite_each_other(tmp_path: Path) -> None:
    path = tmp_path / "litenotebook.sqlite3"
    build_db(path, 8)
    conn = connect(path)
    try:
        first = dbmod.backup_before_migration(conn, 8, LATEST)
        second = dbmod.backup_before_migration(conn, 8, LATEST)
    finally:
        conn.close()
    assert first and second and first != second
    assert first.is_file() and second.is_file()


def test_database_newer_than_the_app_is_refused_untouched(tmp_path: Path) -> None:
    path = tmp_path / "litenotebook.sqlite3"
    build_db(path, LATEST - 1)
    conn = connect(path)
    try:
        migrate(conn)
        conn.execute(f"PRAGMA user_version = {LATEST + 1}")
        with pytest.raises(MigrationError, match="newer"):
            migrate(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == LATEST + 1
    finally:
        conn.close()


def test_wal_database_upgrades_and_reopens(
    tmp_path: Path, fresh_schema: set[tuple[str, str]]
) -> None:
    """The app runs in WAL mode; an upgrade, close and reopen keeps everything."""
    path = tmp_path / "litenotebook.sqlite3"
    counts = build_db(path, 9)
    conn = connect(path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        migrate(conn)
    finally:
        conn.close()
    conn = connect(path)
    try:
        assert migrate(conn) == LATEST
        assert (
            conn.execute("SELECT count(*) FROM notes").fetchone()[0] == counts["notes"]
        )
        assert _schema(conn) == fresh_schema
    finally:
        conn.close()
