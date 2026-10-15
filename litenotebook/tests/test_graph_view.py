"""Phase 18: the knowledge graph (Global and Local views over graph memory)."""

from __future__ import annotations

import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder
from graph_fixtures import build_graph_notebook
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.memory import visual
from litenotebook.memory.concepts import extract_concepts
from litenotebook.notebooks import create_notebook
from litenotebook.notes import create_note
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def g(conn, settings, tmp_path):
    ids = build_graph_notebook(conn, settings, tmp_path)
    ids["other"] = create_notebook(conn, "Other").id
    return ids


def ids_of(graph, kind=None):
    return {n["id"] for n in graph["nodes"] if kind is None or n["type"] == kind}


def edge_set(graph):
    return {(e["source"], e["type"], e["target"]) for e in graph["edges"]}


# --- Global graph -------------------------------------------------------------------


def test_default_global_graph_shows_meaningful_types_only(conn, g):
    view = visual.global_graph(conn, g["nb"])
    assert {n["type"] for n in view["nodes"]} == {
        "concept",
        "note",
        "source",
        "conversation",
        "correction",
    }
    assert f"s:{g['atlas']}" in ids_of(view) and f"m:{g['trip_node']}" in ids_of(view)
    assert not view["truncated"] and view["nodes_total"] == len(view["nodes"])


def test_hidden_questions_and_evidence_are_contracted_not_lost(conn, g):
    view = visual.global_graph(conn, g["nb"])
    conv, atlas = f"m:{g['geo_node']}", f"s:{g['atlas']}"
    supported = [
        e for e in view["edges"] if (e["source"], e["target"]) == (conv, atlas)
    ]
    assert supported and supported[0]["type"] == "supported_by"
    assert supported[0]["via"] == ["answer", "evidence"]
    assert supported[0]["weight"] == 2  # two cited answers in that conversation
    quill = f"m:{g['quillport']}"
    assert (conv, "mentions", quill) in edge_set(view)
    # the correction corrects the conversation's (hidden) answer
    fix = f"m:{g['correction_node']}"
    assert (fix, "corrects", conv) in edge_set(view)
    assert "has_correction" in next(
        n["flags"] for n in view["nodes"] if n["id"] == conv
    )


def test_every_view_edge_is_backed_by_stored_data(conn, g):
    """Contracted edges keep the stored relation; no edge type is invented."""
    allowed = set(visual.RELATION_LABELS)
    view = visual.global_graph(conn, g["nb"], set(visual.ALL_TYPES))
    assert {e["type"] for e in view["edges"]} <= allowed
    raw = {
        r[0]
        for r in conn.execute(
            "SELECT relation FROM memory_edges WHERE notebook_id = ?", (g["nb"],)
        )
    }
    record_refs = {"derived_from", "annotates", "linked_to", "tests", "references"}
    assert {e["type"] for e in view["edges"]} <= raw | record_refs


def test_optional_types_and_their_record_edges(conn, g):
    full = visual.global_graph(conn, g["nb"], set(visual.ALL_TYPES))
    edges = edge_set(full)
    assert {"question", "answer", "evidence", "annotation", "quiz"} <= {
        n["type"] for n in full["nodes"]
    }
    ann, alg = f"a:{g['annotation']}", f"s:{g['algebra']}"
    assert (ann, "annotates", alg) in edges
    assert (f"m:{g['revision_note_node']}", "linked_to", ann) in edges
    quiz = f"z:{g['quiz']}"
    assert (quiz, "tests", f"m:{g['quillport']}") in edges
    assert (quiz, "references", f"s:{g['atlas']}") in edges
    assert any(t == "derived_from" for _, t, _ in edges)


@pytest.fixture(autouse=True)
def _revision_node(request, conn):
    if "g" in request.fixturenames:
        g = request.getfixturevalue("g")
        g["revision_note_node"] = conn.execute(
            "SELECT id FROM memory_nodes WHERE key = ?", (f"note:{g['revision_note']}",)
        ).fetchone()[0]


def test_outcomes_and_study_status(conn, g):
    full = visual.global_graph(conn, g["nb"], set(visual.ALL_TYPES))
    flags = {n["id"]: n["flags"] for n in full["nodes"]}
    river = conn.execute(
        "SELECT id FROM memory_nodes WHERE key = ?", (f"answer:{g['river_answer']}",)
    ).fetchone()[0]
    matrix = conn.execute(
        "SELECT id FROM memory_nodes WHERE key = ?", (f"answer:{g['matrix_answer']}",)
    ).fetchone()[0]
    assert {"corrected", "superseded"} <= set(flags[f"m:{river}"])
    assert flags[f"m:{matrix}"] == ["useful"]
    assert "needs_review" in flags[f"m:{g['quillport']}"]
    assert not any("master" in f for fl in flags.values() for f in fl)


def test_filters_orphans_and_minimum_connections(conn, g, settings, tmp_path):
    create_note(conn, g["nb"], "It is", "")  # stopwords only: no concepts
    lonely = conn.execute(
        "SELECT id FROM memory_nodes WHERE label = 'It is'"
    ).fetchone()[0]
    assert f"m:{lonely}" in ids_of(visual.global_graph(conn, g["nb"]))
    hidden = visual.global_graph(conn, g["nb"], hide_orphans=True)
    assert f"m:{lonely}" not in ids_of(hidden)
    strong = visual.global_graph(conn, g["nb"], min_degree=3)
    assert strong["nodes"] and all(n["degree"] >= 3 for n in strong["nodes"])
    only_sources = visual.global_graph(conn, g["nb"], {"source"})
    assert {n["type"] for n in only_sources["nodes"]} == {"source"}
    assert only_sources["edges"] == []


def test_concept_floor_auto_and_explicit(conn, g):
    auto = visual.global_graph(conn, g["nb"])  # < 40 concepts: everything shown
    assert auto["concept_min"] == 0 and auto["concepts_hidden"] == 0
    shared = visual.global_graph(conn, g["nb"], concept_min=2)
    concepts = [n for n in shared["nodes"] if n["type"] == "concept"]
    assert concepts and all(n["degree"] >= 2 for n in concepts)
    assert shared["concepts_hidden"] > 0
    for i in range(45):  # a busy notebook: auto hides single-mention concepts
        create_note(conn, g["nb"], f"Topic {i}", f"zeppelin{i:02d}x alpha{i:02d}y")
    busy = visual.global_graph(conn, g["nb"])
    assert busy["concept_min_auto"] and busy["concept_min"] == 2
    assert busy["concepts_hidden"] >= 90
    assert all(n["degree"] >= 2 for n in busy["nodes"] if n["type"] == "concept")


def test_caps_report_truncation_deterministically(conn, g):
    small = visual.global_graph(conn, g["nb"], max_nodes=5)
    assert len(small["nodes"]) == 5 and small["truncated"]
    assert small["nodes_total"] > 5 and small["limits"]["nodes"] == 5
    degrees = [n["degree"] for n in small["nodes"]]
    assert degrees == sorted(degrees, reverse=True)
    assert small == visual.global_graph(conn, g["nb"], max_nodes=5)
    few = visual.global_graph(conn, g["nb"], max_edges=4)
    assert len(few["edges"]) == 4 and few["edges_truncated"]
    assert few["edges_total"] > 4
    weights = [e["weight"] for e in few["edges"]]
    assert weights == sorted(weights, reverse=True)  # strongest links kept
    huge = visual.global_graph(conn, g["nb"], max_nodes=10**9, max_edges=10**9)
    assert huge["limits"] == {"nodes": visual.MAX_NODES, "edges": visual.MAX_EDGES}


# --- Local graph ----------------------------------------------------------------------


def test_local_graph_depths_and_centre(conn, g):
    center = f"m:{g['quillport']}"
    sizes = []
    for depth in (1, 2, 3):
        local = visual.local_graph(conn, g["nb"], center, depth)
        assert local["nodes"][0]["id"] == center and local["nodes"][0]["depth"] == 0
        assert max(n["depth"] for n in local["nodes"]) <= depth
        sizes.append(len(local["nodes"]))
    assert sizes[0] < sizes[1] <= sizes[2]
    one = visual.local_graph(conn, g["nb"], center, 1)
    assert ids_of(one) == {center, f"m:{g['trip_node']}", f"m:{g['geo_node']}"}
    assert visual.local_graph(conn, g["nb"], center, 9)["depth"] == visual.MAX_DEPTH


def test_local_graph_bounds_and_hidden_type_centre(conn, g):
    center = f"m:{g['quillport']}"
    tiny = visual.local_graph(conn, g["nb"], center, 3, max_nodes=4)
    assert len(tiny["nodes"]) == 4 and tiny["truncated"] and tiny["nodes_total"] > 4
    answer = conn.execute(
        "SELECT id FROM memory_nodes WHERE key = ?", (f"answer:{g['capital_answer']}",)
    ).fetchone()[0]
    local = visual.local_graph(conn, g["nb"], f"m:{answer}", 1)  # answers hidden
    assert local["nodes"][0]["type"] == "answer" and len(local["nodes"]) > 1


def test_isolation_and_id_tampering(conn, g):
    other = g["other"]
    for bad in (
        f"m:{g['quillport']}",
        f"s:{g['atlas']}",
        f"a:{g['annotation']}",
        f"z:{g['quiz']}",
    ):
        assert visual.node_kind(conn, other, bad) is None
        assert visual.local_graph(conn, other, bad, 2) is None
        assert visual.node_detail(conn, other, bad) is None
    for junk in ("m:abc", "x:1", "m:-1", "m:1;drop", "m:" + "9" * 40, "", "s:", ":1"):
        assert visual.node_kind(conn, g["nb"], junk) is None
    assert visual.global_graph(conn, other)["nodes"] == []
    assert visual.search(conn, other, "quill") == []


# --- inspector, navigation, search ----------------------------------------------------


def test_inspector_detail_and_actions(conn, g):
    nb = g["nb"]
    concept = visual.node_detail(conn, nb, f"m:{g['quillport']}")
    assert concept["flags"] == ["needs_review"] and concept["meta"]["study"]
    assert set(concept["connections"]) == {"note", "conversation", "quiz"}
    note = visual.node_detail(conn, nb, f"m:{g['trip_node']}")
    assert note["actions"][0] == {
        "label": "Open note",
        "href": f"/notebooks/{nb}/notes/{g['trip_note']}",
    }
    conv = visual.node_detail(conn, nb, f"m:{g['geo_node']}")
    assert conv["actions"][0]["href"] == f"/notebooks/{nb}/chat/{g['geo_conv']}"
    source = visual.node_detail(conn, nb, f"s:{g['atlas']}")
    assert [a["href"] for a in source["actions"]] == [
        f"/notebooks/{nb}/sources/{g['atlas']}/read",
        f"/notebooks/{nb}/sources/{g['atlas']}",
    ]
    ann = visual.node_detail(conn, nb, f"a:{g['annotation']}")
    assert ann["actions"][0]["href"].endswith(
        f"/sources/{g['algebra']}/read?annotation={g['annotation']}"
    )
    quiz = visual.node_detail(conn, nb, f"z:{g['quiz']}")
    hrefs = [a["href"] for a in quiz["actions"]]
    assert hrefs[0].endswith(f"/study/quizzes/{g['quiz']}")
    assert hrefs[1].endswith(f"/study/attempts/{g['attempt']}/review")
    fix = visual.node_detail(conn, nb, f"m:{g['correction_node']}")
    assert fix["actions"][0]["href"].startswith(
        f"/notebooks/{nb}/chat/{g['geo_conv']}#"
    )
    assert len(fix["snippet"]) <= visual.SNIPPET_CHARS


def test_evidence_jumps_to_the_cited_passage(conn, g):
    full = visual.global_graph(conn, g["nb"], set(visual.ALL_TYPES))
    evidence = [n for n in full["nodes"] if n["type"] == "evidence"]
    hrefs = [visual.node_detail(conn, g["nb"], n["id"])["actions"][0] for n in evidence]
    assert any("/read?citation=" in a["href"] for a in hrefs)  # chat evidence
    assert any("/read?chunk=" in a["href"] for a in hrefs)  # note evidence


def test_view_building_never_writes(conn, g):
    before = conn.total_changes
    visual.global_graph(conn, g["nb"], set(visual.ALL_TYPES))
    visual.local_graph(conn, g["nb"], f"m:{g['quillport']}", 3)
    visual.node_detail(conn, g["nb"], f"s:{g['atlas']}")
    visual.search(conn, g["nb"], "a")
    assert conn.total_changes == before


# --- web: routes, no AI, security, accessibility --------------------------------------


@pytest.fixture
def web(tmp_path):
    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    conn = open_db(s.db_path)
    ids = build_graph_notebook(conn, s, tmp_path)
    ids["other"] = create_notebook(conn, "Other").id
    evil = create_note(
        conn,
        ids["nb"],
        '<img src=x onerror="alert(1)"> Quillport',
        "<script>x</script>",
    )
    ids["evil_note"] = evil.id
    conn.execute(
        "UPDATE sources SET title = ? WHERE id = ?",
        ('"><svg onload=alert(2)>.pdf', ids["algebra"]),
    )
    conn.commit()
    conn.close()
    calls = {"embed": 0, "llm": 0}

    def factory():
        calls["embed"] += 1
        return FakeEmbedder()

    def llm(request):
        calls["llm"] += 1
        return httpx.Response(500)

    app = create_app(
        s, env={}, embedder_factory=factory, llm_transport=httpx.MockTransport(llm)
    )
    with TestClient(app, base_url=BASE) as c:
        yield {"c": c, "calls": calls, **ids}


def test_graph_routes_and_no_ai(web):
    c, nb = web["c"], web["nb"]
    heavy_before = {m for m in ("torch", "sentence_transformers") if m in sys.modules}
    page = c.get(f"/notebooks/{nb}/memory/graph")
    assert page.status_code == 200 and 'data-mode="global"' in page.text
    view = c.get(f"/notebooks/{nb}/memory/graph/view.json").json()
    assert view["nodes"] and view["mode"] == "global"
    center = f"m:{web['quillport']}"
    local = c.get(
        f"/notebooks/{nb}/memory/graph/local.json?node={center}&depth=2"
    ).json()
    assert local["center"] == center and local["depth"] == 2
    detail = c.get(f"/notebooks/{nb}/memory/graph/node.json?id={center}").json()
    assert detail["label"] == "quillport"
    found = c.get(f"/notebooks/{nb}/memory/graph/find.json?q=QUILL").json()["results"]
    labels = [r["label"] for r in found]
    assert "quillport" in labels and all("quill" in x.lower() for x in labels)
    assert labels.index("quillport") < labels.index("quillport harbour")
    page_local = c.get(f"/notebooks/{nb}/memory/graph?mode=local&node={center}")
    assert f'data-center="{center}"' in page_local.text
    assert web["calls"] == {"embed": 0, "llm": 0}
    heavy_after = {m for m in ("torch", "sentence_transformers") if m in sys.modules}
    assert heavy_after == heavy_before


def test_graph_routes_enforce_notebook_ownership(web):
    c, nb, other = web["c"], web["nb"], web["other"]
    center = f"m:{web['quillport']}"
    for url in (
        f"/notebooks/{other}/memory/graph/local.json?node={center}",
        f"/notebooks/{other}/memory/graph/node.json?id={center}",
        f"/notebooks/{other}/memory/graph/node.json?id=s:{web['atlas']}",
        f"/notebooks/{other}/memory/graph?node={center}",
        f"/notebooks/{nb}/memory/graph/node.json?id=m:999999",
        f"/notebooks/{nb}/memory/graph/local.json?node=s:1%3Bdrop",
        f"/notebooks/{nb}/memory/graph?node=z:0",
        "/notebooks/999/memory/graph/view.json",
    ):
        assert c.get(url).status_code == 404, url
    assert c.get(f"/notebooks/{other}/memory/graph/view.json").json()["nodes"] == []
    assert (
        c.get(f"/notebooks/{other}/memory/graph/find.json?q=quill").json()["results"]
        == []
    )


def test_untrusted_labels_stay_text(web):
    c, nb = web["c"], web["nb"]
    page = c.get(f"/notebooks/{nb}/memory/graph").text
    assert "onerror" not in page and "<svg onload" not in page  # never server-rendered
    view = c.get(
        f"/notebooks/{nb}/memory/graph/view.json?types=note,source,concept"
    ).json()
    labels = {n["label"] for n in view["nodes"]}
    assert '<img src=x onerror="alert(1)"> Quillport' in labels  # data, as text
    assert '"><svg onload=alert(2)>.pdf' in labels
    js = (ROOT / "src/litenotebook/web/static/graph.js").read_text("utf-8")
    assert "innerHTML" not in js.replace("never innerHTML", "")
    assert "insertAdjacentHTML" not in js and "outerHTML" not in js
    assert "eval(" not in js and "new Function" not in js


def test_payloads_do_not_carry_document_text(web):
    c, nb = web["c"], web["nb"]
    body = c.get(
        f"/notebooks/{nb}/memory/graph/view.json?types=" + ",".join(visual.ALL_TYPES)
    ).json()
    for n in body["nodes"]:
        assert set(n) == {"id", "type", "label", "degree", "flags", "meta"}
        assert len(n["label"]) <= visual.LABEL_CHARS
    text = c.get(f"/notebooks/{nb}/memory/graph/view.json").text
    assert "blue glass and salted fish" not in text  # chunk text stays behind Reader
    detail = c.get(
        f"/notebooks/{nb}/memory/graph/node.json?id=m:{web['trip_node']}"
    ).json()
    assert len(detail["snippet"]) <= visual.SNIPPET_CHARS


def test_navigation_targets_open(web):
    c, nb = web["c"], web["nb"]
    for node_id in (
        f"m:{web['trip_node']}",
        f"m:{web['geo_node']}",
        f"s:{web['atlas']}",
        f"a:{web['annotation']}",
        f"z:{web['quiz']}",
        f"m:{web['correction_node']}",
        f"m:{web['quillport']}",
    ):
        detail = c.get(f"/notebooks/{nb}/memory/graph/node.json?id={node_id}").json()
        for action in detail["actions"]:
            r = c.get(action["href"].split("#")[0])
            assert r.status_code == 200, (node_id, action)
    full = c.get(f"/notebooks/{nb}/memory/graph/view.json?types=evidence,source").json()
    ev = next(n for n in full["nodes"] if n["type"] == "evidence")
    href = c.get(f"/notebooks/{nb}/memory/graph/node.json?id={ev['id']}").json()[
        "actions"
    ][0]["href"]
    citation = int(re.search(r"citation=(\d+)", href).group(1))
    target = c.get(f"/notebooks/{nb}/reader/citations/{citation}").json()
    assert target["live"] and target["rects"]  # Reader highlights the passage


class _Controls(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[dict[str, str | None]] = []
        self.labels: set[str] = set()
        self.wrapped = 0
        self.depth_in_label = 0
        self.attrs: list[dict[str, str | None]] = []

    def handle_starttag(self, tag, attrs):  # type: ignore[no-untyped-def]
        a = dict(attrs)
        self.attrs.append({"tag": tag, **a})
        if tag == "label":
            self.depth_in_label += 1
            if a.get("for"):
                self.labels.add(a["for"])
        if tag in ("input", "select"):
            self.inputs.append({**a, "in_label": str(self.depth_in_label > 0)})

    def handle_endtag(self, tag):  # type: ignore[no-untyped-def]
        if tag == "label":
            self.depth_in_label -= 1


def test_graph_page_accessibility(web):
    c, nb = web["c"], web["nb"]
    html = c.get(f"/notebooks/{nb}/memory/graph").text
    start = html.index('id="graph-title"')
    p = _Controls()
    p.feed(html[start:])
    for i in p.inputs:
        if i.get("type") == "hidden":
            continue
        assert i["in_label"] == "True" or i.get("id") in p.labels, i
    stage = next(a for a in p.attrs if a.get("id") == "kg-stage")
    assert stage["tabindex"] == "0" and "Arrow keys" in (stage["aria-label"] or "")
    status = next(a for a in p.attrs if a.get("id") == "kg-status")
    assert status["role"] == "status"
    assert any(a.get("id") == "kg-list-box" and a["tag"] == "details" for a in p.attrs)
    assert any(a.get("id") == "kg-inspector" for a in p.attrs)
    for mode in ("kg-mode-global", "kg-mode-local"):
        assert next(a for a in p.attrs if a.get("id") == mode)["aria-pressed"]
    assert "Graph: move to a connected node" in html  # shortcut help
    assert "Legend" in html


def _js_function(js: str, name: str) -> str:
    start = js.index(f"function {name}(")
    end = js.index("\n  }\n", start)
    return js[start:end]


def test_graph_layout_respects_reduced_motion():
    # The browser suite drives both preferences; this keeps the two paths in the
    # shipped script. Reduced motion: settle in short, yielding chunks, never draw
    # intermediate frames, and move only the dragged node. Otherwise: animate.
    js = (ROOT / "src/litenotebook/web/static/graph.js").read_text("utf-8")
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in js
    warm = _js_function(js, "warm")
    assert warm.index("if (reducedMotion()) { settleQuietly(); return; }") < warm.index(
        "tick()"
    )
    quiet = _js_function(js, "settleQuietly")
    assert "setTimeout(chunk, 0)" in quiet  # yields to input between chunks
    assert "draw()" not in quiet.replace("redraw()", "")  # one draw, once settled
    chunk_ms = re.search(r"QUIET_CHUNK_MS = (\d+)", js)
    assert chunk_ms and int(chunk_ms.group(1)) <= 16
    loop = _js_function(js, "loop")
    assert loop.index("if (quiet) return;") < loop.index("tick()")
    assert "requestAnimationFrame" in loop and "animatedFrames += 1" in loop
    drag = js[js.index('addEventListener("pointermove"') :]
    drag = drag[: drag.index("} else {")]  # the node-drag branch
    assert "if (reducedMotion())" in drag and "else reheat(0.25);" in drag
    assert 'addEventListener("change", onMotion)' in js  # preference changes live


# --- performance --------------------------------------------------------------------


def test_large_synthetic_graph_is_bounded_and_fast(conn, settings):
    nb = create_notebook(conn, "Large")
    concepts, notes = 900, 300
    with conn:
        for i in range(concepts):
            conn.execute(
                "INSERT INTO memory_nodes (notebook_id, key, node_type, label, "
                "content, provenance) VALUES (?, ?, 'concept', ?, ?, 'derived')",
                (nb.id, f"concept:c{i}", f"concept {i}", f"concept {i}"),
            )
        for i in range(notes):
            conn.execute(
                "INSERT INTO memory_nodes (notebook_id, key, node_type, label, "
                "content, provenance) VALUES (?, ?, 'note', ?, '', 'user_note')",
                (nb.id, f"note:{i}", f"note {i}"),
            )
        ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM memory_nodes WHERE notebook_id = ? ORDER BY id",
                (nb.id,),
            )
        ]
        cids, nids = ids[:concepts], ids[concepts:]
        rows = set()
        for j, note in enumerate(nids):
            for k in range(14):
                rows.add((note, cids[(j * 7 + k * k * 13) % concepts]))
        conn.executemany(
            "INSERT INTO memory_edges (notebook_id, from_node_id, to_node_id, "
            "relation, provenance) VALUES (?, ?, ?, 'note_mentions', 'synthetic')",
            [(nb.id, a, b) for a, b in sorted(rows)],
        )
    assert len(rows) > 3000
    started = time.perf_counter()
    view = visual.global_graph(conn, nb.id)
    elapsed = time.perf_counter() - started
    assert view["nodes_in_notebook"] == concepts + notes
    assert len(view["nodes"]) == visual.GLOBAL_NODES and view["truncated"]
    assert len(view["edges"]) <= visual.GLOBAL_EDGES
    assert elapsed < 2.0, elapsed
    local = visual.local_graph(conn, nb.id, f"m:{nids[0]}", 3)
    assert len(local["nodes"]) <= visual.LOCAL_NODES and local["truncated"]
    started = time.perf_counter()
    visual.search(conn, nb.id, "concept 1")
    assert time.perf_counter() - started < 1.0


# --- concept cleanup ----------------------------------------------------------------


def test_markup_never_becomes_a_concept():
    latex = (
        r"The determinant $\det(A)=\frac{a}{b}$ of \mathbf{A} with "
        r"\begin{pmatrix} a \end{pmatrix} rows"
    )
    found = extract_concepts(latex, 20)
    assert not {"det", "frac", "mathbf", "begin", "pmatrix", "end"} & set(
        w for c in found for w in c.split()
    )
    assert "determinant" in found
    web = (
        "Read https://example.org/eigen-values and www.mathworld.com, mail prof@uni.edu"
    )
    words = {w for c in extract_concepts(web, 20) for w in c.split()}
    assert not words & {
        "https",
        "example",
        "org",
        "www",
        "mathworld",
        "com",
        "prof",
        "uni",
        "edu",
    }
    assert extract_concepts("What did we discuss about diagonalising matrix A?") == [
        "diagonalising matrix",
        "diagonalising",
        "matrix",
    ]
