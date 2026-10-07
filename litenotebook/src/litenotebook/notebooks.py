"""Notebook records."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from litenotebook.config import EMBED_DIM, EMBED_MODEL


class NotebookError(ValueError):
    pass


@dataclass(frozen=True)
class Notebook:
    id: int
    name: str
    description: str
    embed_model: str
    embed_dim: int


def _from_row(row: sqlite3.Row) -> Notebook:
    return Notebook(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        embed_model=row["embed_model"],
        embed_dim=row["embed_dim"],
    )


def create_notebook(
    conn: sqlite3.Connection, name: str, description: str = ""
) -> Notebook:
    name = name.strip()
    if not name:
        raise NotebookError("Notebook name cannot be empty")
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


def get_notebook(conn: sqlite3.Connection, notebook_id: int) -> Notebook:
    row = conn.execute(
        "SELECT * FROM notebooks WHERE id = ?", (notebook_id,)
    ).fetchone()
    if row is None:
        raise NotebookError(f"Notebook {notebook_id} not found")
    return _from_row(row)


def get_notebook_by_name(conn: sqlite3.Connection, name: str) -> Notebook:
    row = conn.execute(
        "SELECT * FROM notebooks WHERE name = ?", (name.strip(),)
    ).fetchone()
    if row is None:
        raise NotebookError(f"Notebook {name!r} not found")
    return _from_row(row)


def list_notebooks(conn: sqlite3.Connection) -> list[Notebook]:
    rows = conn.execute("SELECT * FROM notebooks ORDER BY name").fetchall()
    return [_from_row(r) for r in rows]
