"""Build a small, deterministic, synthetic notebook archive for cross-platform
checks (copy it to a Windows machine and import it there — see
docs/windows-test-plan.md).

    python tests/portable_archive.py OUT.zip

Synthetic data only (no private documents). Same inputs → byte-identical
archive on the same OS (``created_on`` in the manifest records the OS).
"""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

FIXED = "2026-01-01T00:00:00.000Z"
FIXED_DT = datetime(2026, 1, 1, tzinfo=UTC)
TIMESTAMP_COLUMNS = {
    "notebooks": ("created_at", "updated_at"),
    "sources": ("created_at", "started_at", "finished_at", "updated_at"),
    "conversations": ("created_at", "updated_at"),
    "messages": ("created_at", "completed_at"),
    "memory_outcomes": ("created_at", "updated_at"),
    "notes": ("created_at", "updated_at"),
    "note_links": ("created_at",),
    "annotations": ("created_at", "updated_at"),
    "reading_state": ("updated_at",),
    "message_memory_refs": ("remembered_at",),
}


def _deterministic_pdf(path: Path, pages: list[str]) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(40, 40, 555, 802), text, fontsize=8)
    doc.set_metadata({})
    doc.save(path, no_new_id=True, garbage=3, deflate=True)
    doc.close()
    return path


def build(out: Path) -> Path:
    import conftest
    from archive_fixture import build_rich_notebook
    from litenotebook.archive.export import export_notebook
    from litenotebook.config import load_settings
    from litenotebook.db import open_db

    conftest.make_pdf = lambda path, pages, **_kw: _deterministic_pdf(path, pages)
    import archive_fixture

    archive_fixture.make_pdf = conftest.make_pdf
    with tempfile.TemporaryDirectory(prefix="ln-portable-") as tmp:
        root = Path(tmp)
        settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(root / "data")})
        conn = open_db(settings.db_path)
        try:
            rich = build_rich_notebook(
                conn, settings, root, name="Portable Ünïcode notes"
            )
            with conn:
                for table, columns in TIMESTAMP_COLUMNS.items():
                    sets = ", ".join(
                        f"{c} = CASE WHEN {c} IS NULL THEN NULL ELSE ? END"
                        for c in columns
                    )
                    conn.execute(f"UPDATE {table} SET {sets}", [FIXED] * len(columns))
                conn.execute(
                    "UPDATE sources SET timings = NULL"
                )  # wall-clock durations
            result = export_notebook(
                conn, settings, rich.notebook_id, out, now=FIXED_DT, export_id="0" * 32
            )
        finally:
            conn.close()
    return result.path


if __name__ == "__main__":
    target = Path(
        sys.argv[1] if len(sys.argv) > 1 else "litenotebook-portable-fixture.zip"
    )
    print(build(target.resolve()))
