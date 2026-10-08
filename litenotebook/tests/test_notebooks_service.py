"""Notebook service: validation, rename, and safe on-disk deletion."""

import pytest

from conftest import make_pdf
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.notebooks import (
    MAX_NAME_LENGTH,
    NotebookError,
    NotebookNotFoundError,
    create_notebook,
    delete_notebook,
    get_notebook,
    get_summary,
    list_summaries,
    rename_notebook,
)


@pytest.mark.parametrize(
    "name", ["", "   ", "x" * (MAX_NAME_LENGTH + 1), "bad\x00name", "tab\tname", "rtl‮"]
)
def test_invalid_names_rejected(conn, name):
    with pytest.raises(NotebookError):
        create_notebook(conn, name)


def test_name_at_max_length_allowed(conn):
    assert create_notebook(conn, "x" * MAX_NAME_LENGTH).name == "x" * MAX_NAME_LENGTH


def test_duplicate_names_are_case_insensitive(conn):
    create_notebook(conn, "Research")
    with pytest.raises(NotebookError, match="already exists"):
        create_notebook(conn, "research ")


def test_rename(conn):
    nb = create_notebook(conn, "Old")
    other = create_notebook(conn, "Other")
    renamed = rename_notebook(conn, nb.id, "  New name ")
    assert renamed.name == "New name"
    assert renamed.updated_at >= nb.updated_at
    assert rename_notebook(conn, nb.id, "new NAME").name == "new NAME"  # own name ok
    with pytest.raises(NotebookError):
        rename_notebook(conn, nb.id, "other")
    with pytest.raises(NotebookError):
        rename_notebook(conn, nb.id, "")
    with pytest.raises(NotebookNotFoundError):
        rename_notebook(conn, 999, "x")
    assert get_notebook(conn, other.id).name == "Other"


def test_summaries_count_sources_and_chunks(conn, settings, embedder, store, tmp_path):
    a = create_notebook(conn, "a")
    create_notebook(conn, "b")
    ingest_pdf(
        conn, settings, a, make_pdf(tmp_path / "x.pdf", ["one", "two"]), embedder, store
    )
    counts = {s.notebook.name: (s.sources, s.chunks) for s in list_summaries(conn)}
    assert counts == {"a": (1, 2), "b": (0, 0)}


def test_delete_cascades_and_removes_only_that_notebooks_files(
    conn, settings, embedder, store, tmp_path
):
    a, b = create_notebook(conn, "a"), create_notebook(conn, "b")
    for nb in (a, b):
        ingest_pdf(
            conn,
            settings,
            nb,
            make_pdf(tmp_path / f"{nb.name}.pdf", ["text"]),
            embedder,
            store,
        )
    assert store.index_path(a.id).exists() and store.index_path(b.id).exists()

    result = delete_notebook(conn, settings, a.id)

    assert (result.sources, result.chunks, result.removed_dir) == (1, 1, True)
    assert not settings.notebook_dir(a.id).exists()
    for table in ("sources", "pages", "chunks"):
        assert (
            conn.execute(
                f"SELECT count(*) FROM {table} WHERE "
                + (
                    "notebook_id = ?"
                    if table != "pages"
                    else "source_id IN (SELECT id FROM sources WHERE notebook_id = ?)"
                ),
                (a.id,),
            ).fetchone()[0]
            == 0
        )
    assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 1
    assert get_summary(conn, b.id).chunks == 1
    assert store.check(b.id).ok


def test_delete_without_files(conn, settings):
    nb = create_notebook(conn, "empty")
    assert delete_notebook(conn, settings, nb.id).removed_dir is False
    with pytest.raises(NotebookNotFoundError):
        delete_notebook(conn, settings, nb.id)


def test_symlinked_notebook_dir_is_unlinked_not_followed(conn, settings, tmp_path):
    nb = create_notebook(conn, "linked")
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete")
    settings.notebooks_dir.mkdir(parents=True)
    settings.notebook_dir(nb.id).symlink_to(outside, target_is_directory=True)

    delete_notebook(conn, settings, nb.id)

    assert not settings.notebook_dir(nb.id).is_symlink()
    assert (outside / "keep.txt").read_text() == "do not delete"


def test_refuses_notebooks_dir_symlinked_outside_data(conn, settings, tmp_path):
    nb = create_notebook(conn, "escape")
    outside = tmp_path / "elsewhere"
    (outside / str(nb.id)).mkdir(parents=True)
    (outside / str(nb.id) / "keep.txt").write_text("x")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.notebooks_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(NotebookError, match="Refusing"):
        delete_notebook(conn, settings, nb.id)
    assert (outside / str(nb.id) / "keep.txt").exists()
