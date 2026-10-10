"""SQLite connection setup and schema migrations.

Migrations are ``migrations/NNNN_name.sql`` files applied in order; the
applied version is tracked in ``PRAGMA user_version``.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{4})_[\w-]+\.sql$")


class MigrationError(RuntimeError):
    pass


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database with WAL, foreign keys and a busy timeout.

    ``check_same_thread=False`` because FastAPI may run a request's dependency
    and handler on different worker threads; each connection is still used by
    only one request at a time.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    # Deleted rows are overwritten instead of lingering in free pages.
    conn.execute("PRAGMA secure_delete = ON")
    return conn


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    migrate(conn)
    secure_fts_delete(conn)
    return conn


FTS_SECURE_DELETE_MIN = (3, 42, 0)


def secure_fts_delete(conn: sqlite3.Connection) -> list[str]:
    """Turn on FTS5 ``secure-delete`` for every full-text table (once).

    Without it FTS5 keeps deleted terms in old index segments until a merge,
    so words of a deleted note or source could still be read from the file.
    The option needs SQLite 3.42+; afterwards older SQLite versions cannot
    read those FTS tables until a ``rebuild``. On older SQLite this is
    skipped (documented in docs/security.md). Returns the tables changed.
    """
    if sqlite3.sqlite_version_info < FTS_SECURE_DELETE_MIN:
        return []
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND sql LIKE 'CREATE VIRTUAL TABLE%USING fts5%'"
        )
    ]
    changed = []
    for table in tables:
        row = conn.execute(
            f"SELECT v FROM \"{table}_config\" WHERE k = 'secure-delete'"
        ).fetchone()
        if row is None or str(row[0]) != "1":
            conn.execute(
                f'INSERT INTO "{table}"("{table}", rank) VALUES (\'secure-delete\', 1)'
            )
            changed.append(table)
    if changed:
        conn.commit()
    return changed


def available_migrations() -> list[tuple[int, Path]]:
    found = []
    for path in MIGRATIONS_DIR.glob("*.sql"):
        match = _MIGRATION_NAME.match(path.name)
        if not match:
            raise MigrationError(f"Badly named migration file: {path.name}")
        found.append((int(match.group(1)), path))
    found.sort()
    versions = [v for v, _ in found]
    if versions != list(range(1, len(found) + 1)):
        raise MigrationError(f"Migrations must be numbered 1..N, got {versions}")
    return found


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations, each in its own transaction. Returns the version.

    Foreign keys are switched off while migrating (it cannot be done inside a
    transaction) so a migration can rebuild a referenced table without the
    DROP cascading into child rows, SQLite's documented procedure for schema
    changes ALTER TABLE cannot make. ``foreign_key_check`` must come back clean
    before each migration commits.
    """
    current = schema_version(conn)
    migrations = available_migrations()
    if current > len(migrations):
        raise MigrationError(
            f"Database schema version {current} is newer than this code "
            f"({len(migrations)})"
        )
    if current == len(migrations):
        return current
    if current > 0:  # an existing database: keep a copy before upgrading it
        backup_before_migration(conn, current, len(migrations))
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for version, path in migrations[current:]:
            sql = path.read_text(encoding="utf-8")
            try:
                conn.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version = {version};")
                violations = conn.execute("PRAGMA foreign_key_check").fetchall()
                if violations:
                    raise MigrationError(
                        f"Migration {path.name} left {len(violations)} foreign key "
                        "violation(s)"
                    )
                conn.commit()
            except (sqlite3.Error, MigrationError) as e:
                conn.rollback()
                if isinstance(e, MigrationError):
                    raise
                raise MigrationError(f"Migration {path.name} failed: {e}") from e
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    return schema_version(conn)


def backup_before_migration(
    conn: sqlite3.Connection, current: int, target: int
) -> Path | None:
    """Copy the database (SQLite online backup, WAL-safe) to
    ``<data>/backups/before-schema-<target>-from-<current>-<time>.sqlite3``
    before migrations change it. An upgrade never resets data; this copy is
    the way back if something goes wrong."""
    from datetime import UTC, datetime

    row = conn.execute("PRAGMA database_list").fetchone()
    file = row[2] if row is not None else ""
    if not file:  # in-memory database
        return None
    folder = Path(file).parent / "backups"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"before-schema-{target}-from-{current}-{stamp}"
    target_path = folder / f"{stem}.sqlite3"
    n = 1
    while target_path.exists():  # never overwrite an earlier safety copy
        n += 1
        target_path = folder / f"{stem}-{n}.sqlite3"
    # Written under a temporary name and renamed when complete, so a crash or
    # a full disk never leaves a half copy that looks like a valid backup.
    partial = target_path.with_name(f".{target_path.name}.partial")
    try:
        dest = sqlite3.connect(partial)
        try:
            conn.backup(dest)
        finally:
            dest.close()
        partial.rename(target_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target_path
