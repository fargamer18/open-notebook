import sqlite3

import pytest

from litenotebook.config import EMBED_DIM, EMBED_MODEL
from litenotebook.db import (
    MigrationError,
    available_migrations,
    connect,
    migrate,
    open_db,
    schema_version,
)
from litenotebook.notebooks import (
    NotebookError,
    create_notebook,
    get_notebook,
    get_notebook_by_name,
    list_notebooks,
)


def _tables(conn):
    return {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")
    }


def test_creates_database_file_and_parent_dirs(settings):
    assert not settings.db_path.exists()
    conn = open_db(settings.db_path)
    assert settings.db_path.exists()
    conn.close()


def test_migration_creates_schema(conn):
    assert {
        "notebooks",
        "sources",
        "pages",
        "chunks",
        "chunks_fts",
        "settings",
    } <= _tables(conn)
    assert schema_version(conn) == len(available_migrations()) == 2


def test_pragmas(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrate_is_idempotent(conn):
    assert migrate(conn) == 2
    assert migrate(conn) == 2


def test_newer_database_is_rejected(settings):
    conn = connect(settings.db_path)
    conn.execute("PRAGMA user_version = 99")
    with pytest.raises(MigrationError):
        migrate(conn)
    conn.close()


def test_failed_migration_rolls_back(settings, monkeypatch, tmp_path):
    bad = tmp_path / "migrations"
    bad.mkdir()
    (bad / "0001_bad.sql").write_text("CREATE TABLE ok (id INTEGER);\nNOT VALID SQL;")
    monkeypatch.setattr("litenotebook.db.MIGRATIONS_DIR", bad)
    conn = connect(settings.db_path)
    with pytest.raises(MigrationError):
        migrate(conn)
    assert schema_version(conn) == 0
    assert "ok" not in _tables(conn)
    conn.close()


def test_create_and_fetch_notebook(conn):
    nb = create_notebook(conn, "  Linear Algebra  ", "notes")
    assert (nb.name, nb.description) == ("Linear Algebra", "notes")
    assert (nb.embed_model, nb.embed_dim) == (EMBED_MODEL, EMBED_DIM)
    assert get_notebook(conn, nb.id) == nb
    assert get_notebook_by_name(conn, "Linear Algebra") == nb
    assert [n.name for n in list_notebooks(conn)] == ["Linear Algebra"]


@pytest.mark.parametrize("name", ["", "   "])
def test_notebook_name_required(conn, name):
    with pytest.raises(NotebookError):
        create_notebook(conn, name)


def test_duplicate_notebook_name(conn):
    create_notebook(conn, "demo")
    with pytest.raises(NotebookError, match="already exists"):
        create_notebook(conn, "demo")


def test_missing_notebook(conn):
    with pytest.raises(NotebookError):
        get_notebook_by_name(conn, "nope")


def test_chunk_token_limit_enforced_by_schema(conn, notebook):
    conn.execute(
        "INSERT INTO sources (id, notebook_id, title, original_filename, stored_path, "
        "mime, sha256, size_bytes, status) VALUES (1, ?, 't', 't.pdf', 'p', "
        "'application/pdf', 'x', 1, 'ready')",
        (notebook.id,),
    )
    conn.execute(
        "INSERT INTO pages (id, source_id, page_no, text, method) "
        "VALUES (1, 1, 1, 'x', 'text')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO chunks (notebook_id, source_id, page_id, page_no, ord, "
            "char_start, char_end, text, n_tokens) "
            "VALUES (?, 1, 1, 1, 0, 0, 1, 'x', 257)",
            (notebook.id,),
        )


def test_fts_triggers_follow_chunk_changes(conn, notebook):
    conn.execute(
        "INSERT INTO sources (id, notebook_id, title, original_filename, stored_path, "
        "mime, sha256, size_bytes, status) VALUES (1, ?, 't', 't.pdf', 'p', "
        "'application/pdf', 'x', 1, 'ready')",
        (notebook.id,),
    )
    conn.execute(
        "INSERT INTO pages (id, source_id, page_no, text, method) "
        "VALUES (1, 1, 1, 'x', 'text')"
    )
    conn.execute(
        "INSERT INTO chunks (id, notebook_id, source_id, page_id, page_no, ord, "
        "char_start, char_end, text, n_tokens) "
        "VALUES (7, ?, 1, 1, 1, 0, 0, 5, 'zebra crossing', 4)",
        (notebook.id,),
    )

    def hits(term):
        return conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (term,)
        ).fetchall()

    assert [r[0] for r in hits("zebra")] == [7]
    conn.execute("UPDATE chunks SET text = 'giraffe' WHERE id = 7")
    assert hits("zebra") == [] and len(hits("giraffe")) == 1
    conn.execute("DELETE FROM sources WHERE id = 1")  # cascades to chunks
    assert hits("giraffe") == []
