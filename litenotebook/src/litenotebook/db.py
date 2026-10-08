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
    return conn


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    migrate(conn)
    return conn


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
    """Apply pending migrations, each in its own transaction. Returns the version."""
    current = schema_version(conn)
    migrations = available_migrations()
    if current > len(migrations):
        raise MigrationError(
            f"Database schema version {current} is newer than this code "
            f"({len(migrations)})"
        )
    for version, path in migrations[current:]:
        sql = path.read_text(encoding="utf-8")
        try:
            conn.executescript(
                f"BEGIN;\n{sql}\nPRAGMA user_version = {version};\nCOMMIT;"
            )
        except sqlite3.Error as e:
            conn.rollback()
            raise MigrationError(f"Migration {path.name} failed: {e}") from e
    return schema_version(conn)
