"""Phase 8: PDF reader — rendering, geometry, citations, annotations, notes."""

import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.chat import CitedChunk, complete_message, get_citation, start_exchange
from litenotebook.config import load_settings
from litenotebook.db import available_migrations, connect, migrate, open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source, ingest_pdf
from litenotebook.notebooks import create_notebook, delete_notebook
from litenotebook.notes import create_note, link_annotation, list_links
from litenotebook.reader.annotations import (
    AnnotationError,
    AnnotationNotFoundError,
    chunk_target,
    citation_target,
    create_annotation,
    delete_annotation,
    get_reading_state,
    list_annotations,
    save_reading_state,
    update_annotation,
    validate_rects,
)
from litenotebook.reader.geometry import match_text, page_geometry, text_offset_hint
from litenotebook.reader.render import (
    PageNotFoundError,
    ReaderError,
    cache_dir,
    nearest_scale,
    open_source,
    render_page,
)
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
ORIGIN = {"Origin": BASE}
HX = {**ORIGIN, "HX-Request": "true"}
REPEAT = "The mirror field focuses sunlight on the receiver tower. "
PAGES = [
    "General introduction about solar energy and the history of mirrors.",
    "Distinctive fact: the Borealis heliostat moves with an accuracy of 0.1 "
    "milliradians, which is remarkably precise for a large outdoor mechanism.",
    REPEAT + "Other words in between that separate the copies. " + REPEAT + "End.",
    "Another fact: the receiver reaches 565 degrees Celsius.",
]


def png_size(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


@pytest.fixture
def src(conn, settings, notebook, tmp_path):
    ingest_pdf(
        conn,
        settings,
        notebook,
        make_pdf(tmp_path / "reader.pdf", PAGES),
        FakeEmbedder(),
        FaissStore(conn, settings),
    )
    return open_source(conn, settings, notebook.id, 1)


def chunk_on(conn, page):
    return conn.execute(
        "SELECT c.id, c.source_id, c.page_no, c.ord, c.text, c.char_start, "
        "p.text AS page_text FROM chunks c JOIN pages p ON p.id = c.page_id "
        "WHERE c.page_no = ? ORDER BY c.id",
        (page,),
    ).fetchone()


# --- rendering ------------------------------------------------------------------------


def test_render_page_png_cache_and_bounds(settings, src):
    assert src.page_count == 4
    path, cached = render_page(settings, src, 2, 1.5)
    assert not cached and path.parent == cache_dir(settings, 1, 1)
    w, h = png_size(path.read_bytes())
    assert (w, h) == (round(595 * 1.5), round(842 * 1.5)) or abs(w - 892) <= 2
    again, cached = render_page(settings, src, 2, 1.49)  # quantised to 1.5
    assert cached and again == path
    big, _ = render_page(settings, src, 2, 2.0)
    assert png_size(big.read_bytes())[0] > w
    for bad in (0, 5, -1):
        with pytest.raises(PageNotFoundError):
            render_page(settings, src, bad, 1.5)
    assert nearest_scale(0.1) == 1.0 and nearest_scale(99) == 3.0


def test_scanned_pdf_renders_without_text(conn, settings, notebook, tmp_path):
    pdf = make_pdf(tmp_path / "scan.pdf", ["SCANNED IMAGE PAGE"], image_only={1})
    from litenotebook.ingest.pipeline import register_source

    register_source(conn, settings, notebook, pdf)  # not ingested: still readable
    reader = open_source(conn, settings, notebook.id, 1)
    path, _ = render_page(settings, reader, 1, 1.0)
    assert png_size(path.read_bytes())[0] > 100
    geometry = page_geometry(reader.path, 1)
    assert not geometry.has_text and geometry.words == []


def test_missing_file_and_other_notebook(conn, settings, notebook, src):
    other = create_notebook(conn, "other")
    from litenotebook.sources import SourceNotFoundError

    with pytest.raises(SourceNotFoundError):
        open_source(conn, settings, other.id, 1)
    (settings.data_dir / src.source.stored_path).unlink()
    with pytest.raises(ReaderError, match="not available"):
        open_source(conn, settings, notebook.id, 1)


def test_source_delete_removes_cache_and_annotations(
    conn, settings, notebook, src, store
):
    render_page(settings, src, 1, 1.5)
    a = create_annotation(
        conn,
        settings,
        notebook.id,
        1,
        page_no=2,
        selected_text="Borealis",
        rects=[[0.1, 0.1, 0.2, 0.2]],
    )
    note = create_note(conn, notebook.id, "Keeps snapshots")
    link_annotation(conn, notebook.id, note.id, a.id)
    save_reading_state(conn, src, 3, 1.2)
    assert cache_dir(settings, notebook.id, 1).is_dir()
    delete_source(conn, settings, notebook.id, 1, store)
    assert not cache_dir(settings, notebook.id, 1).exists()
    for table in (
        "annotations",
        "annotation_rects",
        "reading_state",
        "annotations_fts",
    ):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    [link] = list_links(conn, notebook.id, note.id)
    assert link.link_type == "annotation" and not link.exists and not link.source_live
    assert link.excerpt == "Borealis"


# --- geometry & chunk matching --------------------------------------------------------


def test_words_are_normalised(src):
    g = page_geometry(src.path, 2)
    assert g.has_text and len(g.words) > 10
    for w in g.words:
        assert 0 <= w.x0 < w.x1 <= 1 and 0 <= w.y0 < w.y1 <= 1


def test_every_stored_chunk_maps_to_its_page(conn, src):
    for row in conn.execute(
        "SELECT c.page_no, c.char_start, c.text, p.text FROM chunks c "
        "JOIN pages p ON p.id = c.page_id"
    ):
        rects = match_text(
            page_geometry(src.path, row[0]), row[2], text_offset_hint(row[3], row[1])
        )
        assert rects, row[0]


def test_match_tolerates_whitespace_case_and_hyphenation(src):
    g = page_geometry(src.path, 2)
    variants = [
        "the  Borealis\nheliostat   moves",
        "THE BOREALIS HELIOSTAT MOVES",
        "Borea-\nlis heliostat",
    ]
    for text in variants:
        assert match_text(g, text), text


def test_multiline_and_repeated_passages(tmp_path):
    long = " ".join(f"word{i}" for i in range(120))
    pdf = make_pdf(tmp_path / "ml.pdf", [long, REPEAT * 2])
    g1 = page_geometry(pdf, 1)
    rects = match_text(g1, " ".join(f"word{i}" for i in range(20, 90)))
    assert rects and len(rects) >= 2  # spans several lines
    g2 = page_geometry(pdf, 2)
    first, second = match_text(g2, REPEAT, 0.0), match_text(g2, REPEAT, 0.9)
    assert first and second and first != second


def test_failed_match_returns_nothing(src):
    g = page_geometry(src.path, 2)
    assert match_text(g, "This sentence does not occur anywhere on this page.") is None
    assert match_text(g, "") is None


def test_chunk_target_fallback_when_highlight_unavailable(
    conn, settings, notebook, src
):
    chunk = chunk_on(conn, 2)
    conn.execute(
        "UPDATE chunks SET text = 'zzz completely different text' WHERE id = ?",
        (chunk[0],),
    )
    conn.commit()
    target = chunk_target(conn, settings, notebook.id, chunk[0])
    assert target.page_no == 2 and target.rects is None and target.live
    assert target.reason == "Highlight unavailable for this passage"


# --- citations ------------------------------------------------------------


def _cite(conn, notebook, chunk):
    ex = start_exchange(conn, notebook.id, None, "How accurate is it?")
    complete_message(
        conn,
        ex.assistant_message_id,
        "Very [S1].",
        model="q",
        citations=[
            CitedChunk(
                1, chunk[0], chunk[1], "reader.pdf", chunk[2], chunk[3], chunk[4]
            )
        ],
    )
    return get_citation(conn, notebook.id, 1)


def test_citation_target_live_and_deleted(conn, settings, notebook, src, store):
    citation = _cite(conn, notebook, chunk_on(conn, 2))
    target = citation_target(conn, settings, notebook.id, citation.id)
    assert target.live and target.page_no == 2 and target.source_id == 1
    assert target.rects and target.rects[0][1] < 0.2  # top of page 2
    delete_source(conn, settings, notebook.id, 1, store)
    gone = citation_target(conn, settings, notebook.id, citation.id)
    assert not gone.live and gone.reason == "Original source has been deleted"
    assert gone.text == citation.chunk_text and gone.page_no == 2


# --- annotations ----------------------------------------------------------


@pytest.mark.parametrize(
    "rects",
    [
        [[0.5, 0.5, 0.4, 0.6]],
        [[0, 0, 1.2, 0.5]],
        [[-0.1, 0, 0.2, 0.2]],
        [[1, 2, 3]],
        [["a", 0, 1, 1]],
        "nope",
        [[0.1, 0.1, 0.2, 0.2]] * 201,
    ],
)
def test_rect_validation(rects):
    with pytest.raises(AnnotationError):
        validate_rects(rects)


def test_annotation_lifecycle_and_scoping(conn, settings, notebook, src):
    other = create_notebook(conn, "other")
    h = create_annotation(
        conn,
        settings,
        notebook.id,
        1,
        page_no=2,
        selected_text="Borealis heliostat",
        color="green",
        rects=[[0.1, 0.05, 0.4, 0.07]],
    )
    c = create_annotation(
        conn,
        settings,
        notebook.id,
        1,
        page_no=3,
        annotation_type="comment",
        comment="Page-level thought",
    )
    assert [a.id for a in list_annotations(conn, notebook.id, 1)] == [h.id, c.id]
    assert [a.id for a in list_annotations(conn, notebook.id, 1, 2)] == [h.id]
    assert c.rects == [] and h.rects == [[0.1, 0.05, 0.4, 0.07]]
    edited = update_annotation(
        conn, notebook.id, h.id, comment="Key fact", color="pink"
    )
    assert edited.comment == "Key fact" and edited.color == "pink"
    for bad in [
        dict(color="red"),
        dict(annotation_type="sticker"),
        dict(page_no=9),
        dict(rects=[]),
    ]:
        kw = {
            "page_no": 2,
            "selected_text": "x",
            "rects": [[0.1, 0.1, 0.2, 0.2]],
            **bad,
        }
        with pytest.raises(ReaderError):
            create_annotation(conn, settings, notebook.id, 1, **kw)
    with pytest.raises(AnnotationError):
        create_annotation(
            conn, settings, notebook.id, 1, page_no=2, annotation_type="comment"
        )
    with pytest.raises(AnnotationError):
        create_annotation(
            conn,
            settings,
            notebook.id,
            1,
            page_no=2,
            annotation_type="comment",
            comment="x" * 10001,
        )
    with pytest.raises(AnnotationNotFoundError):
        update_annotation(conn, other.id, h.id, comment="hijack")
    with pytest.raises(AnnotationNotFoundError):
        delete_annotation(conn, other.id, h.id)
    assert list_annotations(conn, other.id) == []
    delete_annotation(conn, notebook.id, h.id)
    assert conn.execute("SELECT count(*) FROM annotation_rects").fetchone()[0] == 0
    hits = conn.execute(
        "SELECT rowid FROM annotations_fts WHERE annotations_fts MATCH 'thought'"
    ).fetchall()
    assert [r[0] for r in hits] == [c.id]


def test_reading_state_bounds(conn, notebook, src):
    assert get_reading_state(conn, notebook.id, 1) == (1, 1.0)
    assert save_reading_state(conn, src, 3, 9.0) == (3, 4.0)
    assert save_reading_state(conn, src, 2, 0.01) == (2, 0.25)
    assert get_reading_state(conn, notebook.id, 1) == (2, 0.25)
    with pytest.raises(PageNotFoundError):
        save_reading_state(conn, src, 99, 1.0)


def test_notebook_delete_cascades_reader_data(conn, settings, notebook, src):
    create_annotation(
        conn,
        settings,
        notebook.id,
        1,
        page_no=1,
        selected_text="x",
        rects=[[0.1, 0.1, 0.2, 0.2]],
    )
    save_reading_state(conn, src, 2, 1.0)
    render_page(settings, src, 1, 1.0)
    delete_notebook(conn, settings, notebook.id)
    for table in ("annotations", "annotation_rects", "reading_state"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    assert not settings.notebook_dir(notebook.id).exists()


# --- migration ------------------------------------------------------------------------


def test_migration_0008_preserves_populated_schema7_db(settings, tmp_path, monkeypatch):
    from litenotebook import db

    partial = tmp_path / "m7"
    partial.mkdir()
    for _v, path in available_migrations()[:7]:
        shutil.copy(path, partial / path.name)
    conn = connect(settings.db_path)
    with monkeypatch.context() as m:
        m.setattr(db, "MIGRATIONS_DIR", partial)
        assert migrate(conn) == 7
    nb = create_notebook(conn, "Maths")
    store = FaissStore(conn, settings)
    ingest_pdf(
        conn, settings, nb, make_pdf(tmp_path / "m.pdf", PAGES), FakeEmbedder(), store
    )
    note = create_note(conn, nb.id, "Review", "Linear algebra")
    chunk = chunk_on(conn, 2)
    # Schema-7 rows written with raw SQL (current code knows annotation_id);
    # an id gap proves ids are preserved, not renumbered.
    for link_id, key in ((3, f"chunk:{chunk[0]}"), (7, "memory:concept:x")):
        conn.execute(
            "INSERT INTO note_links (id, note_id, link_type, target_key, source_id, "
            "chunk_id, memory_key, label, excerpt, page_no, chunk_ord, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'label', 'excerpt', 2, 0, '{}')",
            (
                link_id,
                note.id,
                "source_chunk" if link_id == 3 else "memory_node",
                key,
                chunk[1] if link_id == 3 else None,
                chunk[0] if link_id == 3 else None,
                None if link_id == 3 else "concept:x",
            ),
        )
    conn.commit()
    links_before = conn.execute("SELECT * FROM note_links ORDER BY id").fetchall()
    counts = {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in (
            "notebooks",
            "sources",
            "chunks",
            "notes",
            "note_links",
            "memory_nodes",
            "memory_edges",
        )
    }

    assert migrate(conn) == len(available_migrations())
    assert {
        t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in counts
    } == counts
    after = conn.execute("SELECT * FROM note_links ORDER BY id").fetchall()
    assert [dict(r) | {"annotation_id": None} for r in links_before] == [
        dict(r) for r in after
    ]
    assert store.check(nb.id).ok
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


# --- web ------------------------------------------------------------------


@pytest.fixture
def web(tmp_path):
    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    conn = open_db(s.db_path)
    nb = create_notebook(conn, "Research")
    ingest_pdf(
        conn,
        s,
        nb,
        make_pdf(tmp_path / "reader.pdf", PAGES),
        FakeEmbedder(),
        FaissStore(conn, s),
    )
    citation = _cite(conn, nb, chunk_on(conn, 2))
    other = create_notebook(conn, "Other")
    conn.close()
    calls = []

    def factory():
        calls.append(1)
        return FakeEmbedder()

    app = create_app(
        s,
        env={},
        embedder_factory=factory,
        llm_transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": []})
        ),
    )
    with TestClient(app, base_url=BASE) as c:
        yield {
            "c": c,
            "s": s,
            "nb": nb.id,
            "other": other.id,
            "citation": citation,
            "calls": calls,
        }


def make_highlight(c, nb, **extra):
    fields = {
        "page": 2,
        "kind": "highlight",
        "text": "Borealis heliostat",
        "color": "yellow",
        "rects": json.dumps([[0.1, 0.05, 0.4, 0.07]]),
        **extra,
    }
    return c.post(f"/notebooks/{nb}/sources/1/annotations", data=fields, headers=ORIGIN)


def test_reader_page_and_controls(web):
    c, nb = web["c"], web["nb"]
    page = c.get(f"/notebooks/{nb}/sources/1/read")
    assert page.status_code == 200
    for needle in (
        'id="r-prev"',
        'id="r-next"',
        'id="r-page"',
        'id="r-zoom-in"',
        'id="r-fit"',
        'data-pages="4"',
        ">Annotations<",
        ">Notes<",
        ">Chat<",
        '<script src="/static/reader.js" defer></script>',
    ):
        assert needle in page.text, needle
    assert "stored_path" not in page.text and "/sources/1.pdf" not in page.text
    img = c.get(f"/notebooks/{nb}/sources/1/pages/1/render?scale=1")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert img.headers["x-render-cache"] == "miss"
    assert (
        c.get(f"/notebooks/{nb}/sources/1/pages/1/render?scale=1").headers[
            "x-render-cache"
        ]
        == "hit"
    )
    assert "private" in img.headers["cache-control"]
    assert c.get(f"/notebooks/{nb}/sources/1/pages/7/render").status_code == 404
    assert c.get(f"/notebooks/{nb}/sources/1/pages/abc/render").status_code == 422
    text = c.get(f"/notebooks/{nb}/sources/1/pages/2/text").json()
    assert text["has_text"] and text["method"] == "text" and text["words"][0]["t"]


def test_reader_isolation(web):
    c, other = web["c"], web["other"]
    for url in (
        f"/notebooks/{other}/sources/1/read",
        f"/notebooks/{other}/sources/1/pages/1/render",
        f"/notebooks/{other}/sources/1/pages/1/text",
        f"/notebooks/{other}/sources/1/annotations",
        f"/notebooks/{other}/sources/1/chunks/1/highlight",
        f"/notebooks/{other}/reader/citations/1",
        "/notebooks/1/sources/999/read",
        "/notebooks/1/sources/999/pages/1/render",
    ):
        assert c.get(url).status_code == 404, url
    assert make_highlight(c, other).status_code == 404
    assert (
        c.post(
            f"/notebooks/{other}/sources/1/reading-state",
            data={"page": 2},
            headers=ORIGIN,
        ).status_code
        == 404
    )


def test_citation_opens_reader_at_page_with_highlight(web):
    c, nb, cit = web["c"], web["nb"], web["citation"]
    evidence = c.get(f"/notebooks/{nb}/chat/citations/{cit.id}", headers=HX).text
    assert f"/notebooks/{nb}/sources/1/read?citation={cit.id}" in evidence
    reader = c.get(f"/notebooks/{nb}/sources/1/read?citation={cit.id}").text
    assert 'data-page="2"' in reader and 'data-focus-kind="citation"' in reader
    target = c.get(f"/notebooks/{nb}/reader/citations/{cit.id}").json()
    assert target["live"] and target["page"] == 2 and target["rects"]


def test_deleted_source_disables_open_in_pdf(web):
    c, nb, cit = web["c"], web["nb"], web["citation"]
    c.post(f"/notebooks/{nb}/sources/1/delete", data={"confirm": "1"}, headers=ORIGIN)
    evidence = c.get(f"/notebooks/{nb}/chat/citations/{cit.id}", headers=HX).text
    assert "Original source has been deleted" in evidence
    assert "read?citation=" not in evidence and "disabled" in evidence
    assert cit.chunk_text in evidence  # historical snapshot still shown
    target = c.get(f"/notebooks/{nb}/reader/citations/{cit.id}").json()
    assert target["live"] is False
    assert c.get(f"/notebooks/{nb}/sources/1/read").status_code == 404


def test_annotations_web_flow_and_escaping(web):
    c, nb = web["c"], web["nb"]
    created = make_highlight(
        c, nb, comment="<script>alert(1)</script>", text="<img src=x onerror=alert(2)>"
    )
    assert created.status_code == 201
    aid = created.json()["id"]
    panel = c.get(f"/notebooks/{nb}/sources/1/annotations", headers=HX).text
    assert "<script>alert(1)" not in panel and "&lt;script&gt;" in panel
    assert "<img src=x" not in panel
    assert f'data-reader-annotation="{aid}"' in panel
    page2 = c.get(f"/notebooks/{nb}/sources/1/annotations.json?page=2").json()
    assert [a["id"] for a in page2["annotations"]] == [aid]
    assert c.get(f"/notebooks/{nb}/sources/1/annotations.json?page=3").json() == {
        "annotations": []
    }
    edit = c.post(
        f"/notebooks/{nb}/sources/1/annotations/{aid}",
        data={"comment": "Important test passage", "color": "blue"},
        headers=HX,
    )
    assert edit.status_code == 200 and "Important test passage" in edit.text
    assert edit.headers["HX-Trigger"] == "annotations-changed"
    bad = c.post(
        f"/notebooks/{nb}/sources/1/annotations/{aid}",
        data={"color": "#f00"},
        headers=HX,
    )
    assert bad.status_code == 422
    assert make_highlight(c, nb, rects="[[0.9,0.9,0.1,0.1]]").status_code == 422
    assert make_highlight(c, nb, rects="not json").status_code == 422
    assert make_highlight(c, nb, page="9").status_code == 404
    reader = c.get(f"/notebooks/{nb}/sources/1/read?annotation={aid}").text
    assert 'data-page="2"' in reader and 'data-focus-kind="annotation"' in reader
    found = c.get(
        f"/notebooks/{nb}/search?q=important+test+passage",
        headers={"HX-Request": "true", "HX-Target": "search-results"},
    ).text
    assert f"/notebooks/{nb}/sources/1/read?annotation={aid}" in found
    confirm = c.get(f"/notebooks/{nb}/sources/1/annotations/{aid}/delete", headers=HX)
    assert "Delete this annotation?" in confirm.text
    assert (
        c.post(
            f"/notebooks/{nb}/sources/1/annotations/{aid}/delete", headers=HX
        ).status_code
        == 422
    )
    gone = c.post(
        f"/notebooks/{nb}/sources/1/annotations/{aid}/delete",
        data={"confirm": str(aid)},
        headers=HX,
    )
    assert (
        gone.status_code == 200 and gone.headers["HX-Trigger"] == "annotations-changed"
    )
    assert c.get(f"/notebooks/{nb}/sources/1/annotations.json").json() == {
        "annotations": []
    }
    assert make_highlight(c, nb, kind="x").status_code == 422
    assert (
        c.post(
            f"/notebooks/{nb}/sources/1/annotations",
            data={"page": 1, "kind": "highlight", "rects": "[]"},
            headers={"Origin": "http://evil"},
        ).status_code
        == 403
    )


def test_annotation_to_note_and_note_to_pdf(web):
    c, nb = web["c"], web["nb"]
    aid = make_highlight(c, nb, comment="Key passage").json()["id"]
    form = c.get(f"/notebooks/{nb}/notes/add-form?kind=annotation&id={aid}", headers=HX)
    assert 'name="kind" value="annotation"' in form.text
    saved = c.post(
        f"/notebooks/{nb}/notes/add",
        data={"kind": "annotation", "item_id": aid, "note_id": "new"},
        headers=HX,
    )
    assert "Annotation: reader.pdf p.2" in saved.text
    note = c.get(f"/notebooks/{nb}/notes/1").text
    assert f"/notebooks/{nb}/sources/1/read?annotation={aid}" in note
    assert "Borealis heliostat" in note and "Key passage" in note
    chunk_id = re.search(
        r"chunk_id \d+|chunks/(\d+)",
        c.get(
            f"/notebooks/{nb}/search?q=borealis",
            headers={"HX-Request": "true", "HX-Target": "search-results"},
        ).text,
    )
    assert chunk_id
    c.post(f"/notebooks/{nb}/sources/1/delete", data={"confirm": "1"}, headers=ORIGIN)
    after = c.get(f"/notebooks/{nb}/notes/1").text
    assert (
        "Original source has been deleted." in after and "read?annotation" not in after
    )
    assert "Borealis heliostat" in after  # snapshot kept


def test_reading_state_resume_and_citation_override(web):
    c, nb, cit = web["c"], web["nb"], web["citation"]
    assert c.post(
        f"/notebooks/{nb}/sources/1/reading-state",
        data={"page": 4, "zoom": 1.3},
        headers=ORIGIN,
    ).json() == {"page": 4, "zoom": 1.3}
    assert 'data-page="4"' in c.get(f"/notebooks/{nb}/sources/1/read").text
    assert 'data-zoom="1.3"' in c.get(f"/notebooks/{nb}/sources/1/read").text
    assert (
        'data-page="2"'
        in c.get(f"/notebooks/{nb}/sources/1/read?citation={cit.id}").text
    )
    assert 'data-page="3"' in c.get(f"/notebooks/{nb}/sources/1/read?page=3").text
    assert (
        c.post(
            f"/notebooks/{nb}/sources/1/reading-state",
            data={"page": 40},
            headers=ORIGIN,
        ).status_code
        == 404
    )


def test_side_panels_and_open_in_pdf_links(web):
    c, nb = web["c"], web["nb"]
    assert (
        "Linked evidence"
        not in c.get(f"/notebooks/{nb}/reader/panel/notes", headers=HX).text
    )
    chat = c.get(f"/notebooks/{nb}/reader/panel/chat", headers=HX).text
    assert 'data-in-reader="1"' in chat and 'class="chat in-reader"' in chat
    assert (
        c.get(f"/notebooks/{nb}/reader/panel/chat?c=0", headers=HX).status_code == 200
    )
    rows = c.get(f"/notebooks/{nb}?tab=sources").text
    assert f"/notebooks/{nb}/sources/1/read" in rows
    found = c.get(
        f"/notebooks/{nb}/search?q=borealis&mode=sources",
        headers={"HX-Request": "true", "HX-Target": "search-results"},
    ).text
    assert re.search(rf"/notebooks/{nb}/sources/1/read\?chunk=\d+", found)
    js = c.get("/static/reader.js").text
    assert ".innerHTML" not in js and "insertAdjacentHTML" not in js
    assert "textContent" in js


def test_reader_never_starts_the_worker(tmp_path):
    code = f"""
import sys, json, httpx
sys.path.insert(0, {str(Path(__file__).parent)!r})
from fastapi.testclient import TestClient
from conftest import FakeEmbedder, make_pdf
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.notebooks import create_notebook
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.web.app import create_app
from pathlib import Path
s = load_settings({{"LITENOTEBOOK_DATA_DIR": {str(tmp_path / "d")!r}}})
c = open_db(s.db_path)
nb = create_notebook(c, "n")
ingest_pdf(c, s, nb, make_pdf(Path({str(tmp_path)!r}) / "x.pdf", ["Heliostat text."]),
           FakeEmbedder(), FaissStore(c, s))
c.close()
del sys.modules["faiss"]  # ingestion above used FAISS; prove the reader does not
calls = []
def factory():
    calls.append(1)
    raise AssertionError("must not create the embedder")
def llama(r):
    return httpx.Response(200, json={{"data": []}})
app = create_app(s, env={{}}, embedder_factory=factory,
                 llm_transport=httpx.MockTransport(llama))
h = {{"Origin": "{BASE}"}}
b = "/notebooks/1/sources/1"
with TestClient(app, base_url="{BASE}") as cl:
    for url in ("/read", "/pages/1/render", "/pages/1/text", "/annotations",
                "/annotations.json?page=1", "/chunks/1/highlight"):
        assert cl.get(b + url).status_code == 200, url
    a = cl.post(b + "/annotations", headers=h, data={{"page": 1, "kind": "highlight",
        "text": "Heliostat", "rects": json.dumps([[0.1, 0.1, 0.3, 0.2]])}}).json()
    r = cl.post(b + "/reading-state", headers=h, data={{"page": 1}})
    assert r.status_code == 200
    assert cl.post("/notebooks/1/notes/add", headers=h, data={{"kind": "annotation",
        "item_id": a["id"], "note_id": "new"}}, follow_redirects=False).status_code \\
        == 303
    assert cl.get("/notebooks/1/search?q=heliostat&mode=annotations").status_code == 200
    assert cl.get("/notebooks/1/reader/panel/notes").status_code == 200
heavy = ("torch", "sentence_transformers", "transformers", "faiss")
print("HEAVY", [m for m in heavy if m in sys.modules], len(calls))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "HEAVY [] 0" in out.stdout


def test_pdf_with_javascript_and_links_renders_passively(
    conn, settings, notebook, tmp_path
):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Click here for danger")
    page.insert_link(
        {
            "kind": pymupdf.LINK_URI,
            "from": pymupdf.Rect(72, 60, 300, 80),
            "uri": "https://example.invalid/",
        }
    )
    doc.save(tmp_path / "js.pdf")
    doc.close()
    from litenotebook.ingest.pipeline import register_source

    register_source(conn, settings, notebook, tmp_path / "js.pdf")
    reader = open_source(conn, settings, notebook.id, 1)
    path, _ = render_page(settings, reader, 1, 1.0)
    assert png_size(path.read_bytes())[0] > 100
    words = page_geometry(reader.path, 1).words
    assert [w.text for w in words][:2] == ["Click", "here"]
