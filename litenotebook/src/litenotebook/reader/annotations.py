"""Reader annotations (highlights + comments), reading position, and
highlight targets for chunks and [S#] citations.

Annotations belong to their source: deleting the source deletes them (FK
cascade). Anything copied into a note keeps its own snapshot. All inputs are
validated: colours come from a fixed set, rectangles must lie inside the
normalised page (0..1), pages must exist, text lengths are bounded.

No heavy imports here; nothing in the reader starts the embedding worker.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from litenotebook.config import Settings
from litenotebook.reader.render import (
    ReaderError,
    ReaderSource,
    check_page,
    open_source,
)

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
COLORS = ("yellow", "green", "blue", "pink")
TYPES = ("highlight", "comment")
MAX_RECTS = 200
MAX_SELECTED = 5000
MAX_COMMENT = 10000
ZOOM_RANGE = (0.25, 4.0)


class AnnotationError(ReaderError):
    pass


class AnnotationNotFoundError(AnnotationError):
    pass


@dataclass
class Annotation:
    id: int
    notebook_id: int
    source_id: int
    source_filename: str
    page_no: int
    annotation_type: str
    selected_text: str
    comment: str
    color: str
    created_at: str
    updated_at: str
    rects: list[list[float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "page": self.page_no,
            "type": self.annotation_type,
            "text": self.selected_text,
            "comment": self.comment,
            "color": self.color,
            "rects": self.rects,
            "updated_at": self.updated_at,
        }


def validate_rects(raw: Any) -> list[list[float]]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_RECTS:
        raise AnnotationError(f"At most {MAX_RECTS} rectangles are allowed")
    rects = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 4:
            raise AnnotationError("Each rectangle needs four numbers")
        try:
            x0, y0, x1, y1 = (float(v) for v in item)
        except (TypeError, ValueError) as e:
            raise AnnotationError("Rectangle coordinates must be numbers") from e
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise AnnotationError("Rectangles must lie inside the page")
        rects.append([x0, y0, x1, y1])
    return rects


def _clean_text(value: str, limit: int, what: str) -> str:
    value = (value or "").replace("\x00", "").strip()
    if len(value) > limit:
        raise AnnotationError(f"{what} is limited to {limit} characters")
    return value


def _rows_to_annotations(
    conn: sqlite3.Connection, rows: list[sqlite3.Row]
) -> list[Annotation]:
    items = [
        Annotation(
            id=r["id"],
            notebook_id=r["notebook_id"],
            source_id=r["source_id"],
            source_filename=r["original_filename"],
            page_no=r["page_no"],
            annotation_type=r["annotation_type"],
            selected_text=r["selected_text"],
            comment=r["comment"],
            color=r["color"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]
    if items:
        by_id = {a.id: a for a in items}
        marks = ",".join("?" * len(items))
        for r in conn.execute(
            f"SELECT annotation_id, x0, y0, x1, y1 FROM annotation_rects "
            f"WHERE annotation_id IN ({marks}) ORDER BY annotation_id, ordinal",
            list(by_id),
        ):
            by_id[r[0]].rects.append([r[1], r[2], r[3], r[4]])
    return items


_SELECT = (
    "SELECT a.*, s.original_filename FROM annotations a "
    "JOIN sources s ON s.id = a.source_id"
)


def get_annotation(
    conn: sqlite3.Connection, notebook_id: int, annotation_id: int
) -> Annotation:
    rows = conn.execute(
        _SELECT + " WHERE a.id = ? AND a.notebook_id = ? AND s.notebook_id = ?",
        (annotation_id, notebook_id, notebook_id),
    ).fetchall()
    if not rows:
        raise AnnotationNotFoundError(
            f"Annotation {annotation_id} not found in this notebook"
        )
    return _rows_to_annotations(conn, rows)[0]


def list_annotations(
    conn: sqlite3.Connection,
    notebook_id: int,
    source_id: int | None = None,
    page_no: int | None = None,
) -> list[Annotation]:
    sql = _SELECT + " WHERE a.notebook_id = ? AND s.notebook_id = ?"
    params: list[Any] = [notebook_id, notebook_id]
    if source_id is not None:
        sql += " AND a.source_id = ?"
        params.append(source_id)
    if page_no is not None:
        sql += " AND a.page_no = ?"
        params.append(page_no)
    rows = conn.execute(sql + " ORDER BY a.page_no, a.id", params).fetchall()
    return _rows_to_annotations(conn, rows)


def create_annotation(
    conn: sqlite3.Connection,
    settings: Settings,
    notebook_id: int,
    source_id: int,
    *,
    page_no: int,
    annotation_type: str = "highlight",
    selected_text: str = "",
    comment: str = "",
    color: str = "yellow",
    rects: Any = None,
) -> Annotation:
    reader = open_source(conn, settings, notebook_id, source_id)
    page_no = check_page(reader, page_no)
    if annotation_type not in TYPES:
        raise AnnotationError("Unknown annotation type")
    if color not in COLORS:
        raise AnnotationError("Unknown colour")
    clean_rects = validate_rects(rects)
    text = _clean_text(selected_text, MAX_SELECTED, "The selected text")
    note = _clean_text(comment, MAX_COMMENT, "A comment")
    if annotation_type == "highlight" and not clean_rects:
        raise AnnotationError("Select some text to highlight")
    if annotation_type == "comment" and not (note or text):
        raise AnnotationError("Write a comment")
    with conn:
        cur = conn.execute(
            "INSERT INTO annotations (notebook_id, source_id, page_no, "
            "annotation_type, "
            "selected_text, comment, color) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (notebook_id, source_id, page_no, annotation_type, text, note, color),
        )
        assert cur.lastrowid is not None
        conn.executemany(
            "INSERT INTO annotation_rects (annotation_id, page_no, x0, y0, x1, y1, "
            "ordinal) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(cur.lastrowid, page_no, *r, i) for i, r in enumerate(clean_rects)],
        )
    return get_annotation(conn, notebook_id, cur.lastrowid)


def update_annotation(
    conn: sqlite3.Connection,
    notebook_id: int,
    annotation_id: int,
    *,
    comment: str | None = None,
    color: str | None = None,
) -> Annotation:
    current = get_annotation(conn, notebook_id, annotation_id)
    new_comment = (
        _clean_text(comment, MAX_COMMENT, "A comment")
        if comment is not None
        else current.comment
    )
    new_color = color if color is not None else current.color
    if new_color not in COLORS:
        raise AnnotationError("Unknown colour")
    with conn:
        conn.execute(
            f"UPDATE annotations SET comment = ?, color = ?, updated_at = {NOW_SQL} "
            "WHERE id = ? AND notebook_id = ?",
            (new_comment, new_color, annotation_id, notebook_id),
        )
    return get_annotation(conn, notebook_id, annotation_id)


def delete_annotation(
    conn: sqlite3.Connection, notebook_id: int, annotation_id: int
) -> Annotation:
    current = get_annotation(conn, notebook_id, annotation_id)
    with conn:
        conn.execute(
            "DELETE FROM annotations WHERE id = ? AND notebook_id = ?",
            (annotation_id, notebook_id),
        )
    return current


# --- reading position -------------------------------------------------------------


def get_reading_state(
    conn: sqlite3.Connection, notebook_id: int, source_id: int
) -> tuple[int, float]:
    row = conn.execute(
        "SELECT current_page, zoom FROM reading_state WHERE source_id = ? "
        "AND notebook_id = ?",
        (source_id, notebook_id),
    ).fetchone()
    return (int(row[0]), float(row[1])) if row else (1, 1.0)


def save_reading_state(
    conn: sqlite3.Connection,
    reader: ReaderSource,
    page_no: int,
    zoom: float,
) -> tuple[int, float]:
    page_no = check_page(reader, page_no)
    zoom = round(min(max(float(zoom), ZOOM_RANGE[0]), ZOOM_RANGE[1]), 3)
    with conn:
        conn.execute(
            "INSERT INTO reading_state (source_id, notebook_id, current_page, zoom) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(source_id) DO UPDATE SET "
            "current_page = excluded.current_page, zoom = excluded.zoom, "
            f"updated_at = {NOW_SQL}",
            (reader.source.id, reader.source.notebook_id, page_no, zoom),
        )
    return page_no, zoom


# --- highlight targets ------------------------------------------------------------


@dataclass
class HighlightTarget:
    source_id: int | None
    page_no: int
    text: str
    rects: list[list[float]] | None
    live: bool  # the original source and chunk still exist
    reason: str = ""


def chunk_target(
    conn: sqlite3.Connection, settings: Settings, notebook_id: int, chunk_id: int
) -> HighlightTarget:
    from litenotebook.reader.geometry import match_text, page_geometry, text_offset_hint

    row = conn.execute(
        "SELECT c.source_id, c.page_no, c.char_start, c.text, p.text AS page_text "
        "FROM chunks c JOIN pages p ON p.id = c.page_id "
        "JOIN sources s ON s.id = c.source_id "
        "WHERE c.id = ? AND c.notebook_id = ? AND s.notebook_id = ?",
        (chunk_id, notebook_id, notebook_id),
    ).fetchone()
    if row is None:
        raise AnnotationNotFoundError("That passage is no longer in this notebook")
    reader = open_source(conn, settings, notebook_id, row["source_id"])
    geometry = page_geometry(reader.path, check_page(reader, row["page_no"]))
    rects = match_text(
        geometry, row["text"], text_offset_hint(row["page_text"], row["char_start"])
    )
    return HighlightTarget(
        source_id=row["source_id"],
        page_no=row["page_no"],
        text=row["text"],
        rects=rects,
        live=True,
        reason="" if rects else "Highlight unavailable for this passage",
    )


def citation_target(
    conn: sqlite3.Connection, settings: Settings, notebook_id: int, citation_id: int
) -> HighlightTarget:
    """Where an [S#] points, from persisted citation metadata (never the
    model's text). A deleted source yields the snapshot with live=False."""
    from litenotebook.chat import get_citation

    citation = get_citation(conn, notebook_id, citation_id)
    if not citation.chunk_exists:
        return HighlightTarget(
            source_id=None,
            page_no=citation.page_no,
            text=citation.chunk_text,
            rects=None,
            live=False,
            reason="Original source has been deleted",
        )
    return chunk_target(conn, settings, notebook_id, citation.chunk_id)
