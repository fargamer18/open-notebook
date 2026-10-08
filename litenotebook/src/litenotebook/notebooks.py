"""Notebook records: create, rename, list, delete (with on-disk cleanup)."""

from __future__ import annotations

import logging
import sqlite3
import unicodedata
from dataclasses import dataclass

from litenotebook.config import EMBED_DIM, EMBED_MODEL, Settings
from litenotebook.platform_utils.filesystem import remove_tree

log = logging.getLogger(__name__)

MAX_NAME_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 1000


class NotebookError(ValueError):
    pass


class NotebookNotFoundError(NotebookError):
    pass


@dataclass(frozen=True)
class Notebook:
    id: int
    name: str
    description: str
    embed_model: str
    embed_dim: int
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class NotebookSummary:
    notebook: Notebook
    sources: int
    chunks: int


@dataclass(frozen=True)
class DeleteResult:
    notebook: Notebook
    sources: int
    chunks: int
    removed_dir: bool


def _from_row(row: sqlite3.Row) -> Notebook:
    return Notebook(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        embed_model=row["embed_model"],
        embed_dim=row["embed_dim"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def validate_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise NotebookError("Notebook name cannot be empty")
    if len(name) > MAX_NAME_LENGTH:
        raise NotebookError(
            f"Notebook name must be at most {MAX_NAME_LENGTH} characters"
        )
    if any(unicodedata.category(ch) in ("Cc", "Cf") for ch in name):
        raise NotebookError("Notebook name cannot contain control characters")
    return name


def _validate_description(description: str) -> str:
    description = description.strip()
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise NotebookError(
            f"Description must be at most {MAX_DESCRIPTION_LENGTH} characters"
        )
    return description


def _ensure_unique(conn: sqlite3.Connection, name: str, exclude_id: int | None) -> None:
    """Names are unique case-insensitively ("Demo" and "demo" would be confusing)."""
    folded = name.casefold()
    for row in conn.execute("SELECT id, name FROM notebooks"):
        if row["id"] != exclude_id and row["name"].casefold() == folded:
            raise NotebookError(f"A notebook named {row['name']!r} already exists")


def create_notebook(
    conn: sqlite3.Connection, name: str, description: str = ""
) -> Notebook:
    name = validate_name(name)
    description = _validate_description(description)
    _ensure_unique(conn, name, exclude_id=None)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO notebooks (name, description, embed_model, embed_dim) "
                "VALUES (?, ?, ?, ?)",
                (name, description, EMBED_MODEL, EMBED_DIM),
            )
    except sqlite3.IntegrityError as e:
        raise NotebookError(f"Notebook {name!r} already exists") from e
    assert cur.lastrowid is not None
    return get_notebook(conn, cur.lastrowid)


def rename_notebook(conn: sqlite3.Connection, notebook_id: int, name: str) -> Notebook:
    get_notebook(conn, notebook_id)
    name = validate_name(name)
    _ensure_unique(conn, name, exclude_id=notebook_id)
    try:
        with conn:
            conn.execute(
                "UPDATE notebooks SET name = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (name, notebook_id),
            )
    except sqlite3.IntegrityError as e:
        raise NotebookError(f"Notebook {name!r} already exists") from e
    return get_notebook(conn, notebook_id)


def get_notebook(conn: sqlite3.Connection, notebook_id: int) -> Notebook:
    row = conn.execute(
        "SELECT * FROM notebooks WHERE id = ?", (notebook_id,)
    ).fetchone()
    if row is None:
        raise NotebookNotFoundError(f"Notebook {notebook_id} not found")
    return _from_row(row)


def get_notebook_by_name(conn: sqlite3.Connection, name: str) -> Notebook:
    row = conn.execute(
        "SELECT * FROM notebooks WHERE name = ?", (name.strip(),)
    ).fetchone()
    if row is None:
        raise NotebookNotFoundError(f"Notebook {name!r} not found")
    return _from_row(row)


def list_notebooks(conn: sqlite3.Connection) -> list[Notebook]:
    rows = conn.execute("SELECT * FROM notebooks ORDER BY name").fetchall()
    return [_from_row(r) for r in rows]


_SUMMARY_SQL = (
    "SELECT n.*, "
    "(SELECT count(*) FROM sources s WHERE s.notebook_id = n.id) AS n_sources, "
    "(SELECT count(*) FROM chunks c WHERE c.notebook_id = n.id) AS n_chunks "
    "FROM notebooks n"
)


def list_summaries(conn: sqlite3.Connection) -> list[NotebookSummary]:
    rows = conn.execute(_SUMMARY_SQL + " ORDER BY n.name COLLATE NOCASE").fetchall()
    return [NotebookSummary(_from_row(r), r["n_sources"], r["n_chunks"]) for r in rows]


def get_summary(conn: sqlite3.Connection, notebook_id: int) -> NotebookSummary:
    row = conn.execute(_SUMMARY_SQL + " WHERE n.id = ?", (notebook_id,)).fetchone()
    if row is None:
        raise NotebookNotFoundError(f"Notebook {notebook_id} not found")
    return NotebookSummary(_from_row(row), row["n_sources"], row["n_chunks"])


def _remove_notebook_dir(settings: Settings, notebook_id: int) -> bool:
    """Delete ``<data_dir>/notebooks/<id>`` and nothing else.

    The path is built from the integer id only, and every component is checked
    to resolve inside the configured data directory before anything is removed.
    A symlink in place of the notebook directory is unlinked, never followed.
    """
    data_root = settings.data_dir.resolve()
    notebooks_root = settings.notebooks_dir
    target = settings.notebook_dir(int(notebook_id))
    if not target.exists() and not target.is_symlink():
        return False
    if notebooks_root.is_symlink() or not notebooks_root.resolve().is_relative_to(
        data_root
    ):
        raise NotebookError("Refusing to delete: notebooks directory is outside data")
    if target.is_symlink():
        target.unlink()
        return True
    resolved = target.resolve()
    if resolved.parent != notebooks_root.resolve() or resolved.name != str(notebook_id):
        raise NotebookError("Refusing to delete a path outside the notebook directory")
    remove_tree(resolved)  # clears read-only attributes (Windows)
    return True


def delete_notebook(
    conn: sqlite3.Connection, settings: Settings, notebook_id: int
) -> DeleteResult:
    """Delete the notebook (sources/pages/chunks cascade) and its files/FAISS index."""
    summary = get_summary(conn, notebook_id)
    with conn:
        busy = conn.execute(
            "SELECT 1 FROM sources WHERE notebook_id = ? AND status = 'processing'",
            (notebook_id,),
        ).fetchone()
        if busy:
            raise NotebookError(
                "A source in this notebook is being ingested; wait for it to finish"
            )
        conn.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
    try:
        removed = _remove_notebook_dir(settings, notebook_id)
    except OSError as e:  # the DB is authoritative; leftover files are harmless
        log.warning("Notebook %s deleted but its directory remains: %s", notebook_id, e)
        removed = False
    return DeleteResult(summary.notebook, summary.sources, summary.chunks, removed)
