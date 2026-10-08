"""Phase 7: unified search and the memory graph explorer."""

import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.chat import CitedChunk, complete_message, start_exchange
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.embed.worker_client import WorkerEmbedder
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source, ingest_pdf
from litenotebook.memory.explorer import (
    MAX_EDGES,
    MAX_NODES,
    default_center,
    edge_detail,
    search_nodes,
    subgraph,
)
from litenotebook.memory.graph import set_outcome, sync_turn
from litenotebook.notebooks import create_notebook
from litenotebook.notes import create_note, link_chunk
from litenotebook.unified_search import MAX_QUERY_CHARS, clean_query, unified_search
from litenotebook.web.app import create_app
from test_embed_worker import pids, process_gone

BASE = "http://127.0.0.1:5056"
HX_SEARCH = {"HX-Request": "true", "HX-Target": "search-results"}
TERM = "heliostat"


def build(conn, settings, nb, tmp_path, term=TERM):
    """One source, one conversation (+memory), one note — all mention ``term``."""
    pdf = make_pdf(
        tmp_path / f"{nb.id}.pdf",
        [f"The {term} tracks the sun across the sky.", "Unrelated page."],
    )
    ingest_pdf(conn, settings, nb, pdf, FakeEmbedder(), FaissStore(conn, settings))
    chunk = conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks "
        "WHERE notebook_id = ? ORDER BY id",
        (nb.id,),
    ).fetchone()
    ex = start_exchange(conn, nb.id, None, f"What does a {term} do?")
    complete_message(
        conn,
        ex.assistant_message_id,
        f"A {term} tracks the sun [S1].",
        model="qwen",
        citations=[
            CitedChunk(
                1, chunk[0], chunk[1], f"{nb.id}.pdf", chunk[2], chunk[3], chunk[4]
            )
        ],
    )
    sync_turn(conn, ex.assistant_message_id)
    note = create_note(conn, nb.id, f"{term.title()} notes", f"My {term} observations.")
    link_chunk(conn, nb.id, note.id, chunk[0])
    return {"chunk": chunk, "ex": ex, "note": note}


@pytest.fixture
def data(conn, settings, notebook, tmp_path):
    return build(conn, settings, notebook, tmp_path)


# --- unified search service -----------------------------------------------------


def test_all_mode_finds_every_category(conn, settings, notebook, data):
    r = unified_search(conn, settings, notebook.id, TERM)
    assert set(r.sections) == {"sources", "notes", "annotations", "chats", "memory"}
    assert all(r.sections[s] for s in ("sources", "notes", "chats", "memory"))
    assert r.sections["annotations"] == []  # none made in this test
    src = r.sections["sources"][0]
    assert src.title == f"{notebook.id}.pdf" and src.meta["match"] == "lexical"
    assert "<mark>" in str(src.snippet) and src.url.endswith(
        f"/chat/chunks/{data['chunk'][0]}"
    )
    assert (
        r.sections["notes"][0].url
        == f"/notebooks/{notebook.id}/notes/{data['note'].id}"
    )
    chat_urls = [h.url for h in r.sections["chats"]]
    assert f"/notebooks/{notebook.id}/chat/{data['ex'].conversation.id}" in chat_urls
    assert any(u.endswith(f"#msg-{data['ex'].assistant_message_id}") for u in chat_urls)
    assert all("/memory/nodes/" in h.url for h in r.sections["memory"])


@pytest.mark.parametrize("mode", ["sources", "notes", "chats", "memory"])
def test_single_section_modes(conn, settings, notebook, data, mode):
    r = unified_search(conn, settings, notebook.id, TERM, mode=mode)
    assert list(r.sections) == [mode] and r.sections[mode]


@pytest.mark.parametrize(
    "query",
    [
        '"unbalanced',
        "(((",
        "a AND OR NOT b",
        "NEAR(x y)",
        "col:value*",
        "^start",
        "-- drop table chunks;",
        "ünïcødé 漢字 🔭",
        "\x00\x02\x03 nul",
        "x" * 5000,
        "<script>alert(1)</script>",
        "   ",
    ],
)
def test_malformed_queries_never_fail(conn, settings, notebook, data, query):
    r = unified_search(conn, settings, notebook.id, query)
    assert len(r.query) <= MAX_QUERY_CHARS and not r.errors
    for hits in r.sections.values():
        for h in hits:
            assert "<script>" not in str(h.snippet)


def test_unicode_query_matches(conn, settings, notebook):
    create_note(conn, notebook.id, "Ünïcødé école", "Réflexions sur l'école.")
    r = unified_search(conn, settings, notebook.id, "école", mode="notes")
    assert r.sections["notes"] and "<mark>" in str(r.sections["notes"][0].snippet)


def test_snippets_escape_document_html(conn, settings, notebook):
    create_note(conn, notebook.id, "Danger", f"{TERM} <img src=x onerror=alert(1)>")
    hit = unified_search(conn, settings, notebook.id, TERM, mode="notes").sections[
        "notes"
    ][0]
    assert "<img" not in str(hit.snippet) and "&lt;img" in str(hit.snippet)
    assert clean_query("a\x02b\x03c") == "a b c"  # our markers cannot be injected


def test_search_is_notebook_scoped(conn, settings, notebook, data, tmp_path):
    other = create_notebook(conn, "other")
    r = unified_search(conn, settings, other.id, TERM)
    assert r.total == 0
    (tmp_path / "o").mkdir()
    build(conn, settings, other, tmp_path / "o", term="anemometer")
    mine = unified_search(conn, settings, notebook.id, "anemometer")
    assert mine.total == 0


def test_chats_search_skips_failed_messages_and_prompts(conn, settings, notebook):
    from litenotebook.chat import end_message

    ex = start_exchange(conn, notebook.id, None, "zebracorn question?")
    end_message(
        conn,
        ex.assistant_message_id,
        "failed",
        "zebracorn error text",
        content="zebracorn partial",
    )
    r = unified_search(conn, settings, notebook.id, "zebracorn", mode="chats")
    roles = [h.meta["role"] for h in r.sections["chats"]]
    assert "assistant" not in roles  # failed answer not searchable
    assert "user" in roles
    sysr = unified_search(
        conn, settings, notebook.id, "untrusted documents", mode="chats"
    )
    assert sysr.sections["chats"] == []  # the system prompt is never stored


def test_hybrid_uses_the_embedder_and_lexical_does_not(conn, settings, notebook, data):
    calls = []

    class Spy(FakeEmbedder):
        def encode(self, texts):
            calls.append(texts)
            return super().encode(texts)

    unified_search(conn, settings, notebook.id, TERM, engine="lexical", embedder=Spy())
    assert calls == []
    r = unified_search(
        conn,
        settings,
        notebook.id,
        TERM,
        engine="hybrid",
        embedder=Spy(),
        mode="sources",
    )
    assert calls and r.sections["sources"][0].meta["match"] in (
        "semantic + lexical",
        "semantic",
        "lexical",
    )
    assert "fused" in r.sections["sources"][0].meta


def test_semantic_failure_is_reported_not_raised(conn, settings, notebook, data):
    from litenotebook.embed.embedder import EmbeddingError

    class Broken(FakeEmbedder):
        def encode(self, texts):
            raise EmbeddingError("worker down")

    r = unified_search(
        conn, settings, notebook.id, TERM, engine="hybrid", embedder=Broken()
    )
    assert "Semantic search is unavailable" in r.errors["sources"]
    assert r.sections["notes"]  # other sections still answered


# --- graph explorer ------------------------------------------------------------------


def node_id(conn, key):
    return conn.execute("SELECT id FROM memory_nodes WHERE key = ?", (key,)).fetchone()[
        0
    ]


def test_subgraph_depths_and_relations(conn, notebook, data):
    center = node_id(conn, f"concept:{TERM}")
    one = subgraph(conn, notebook.id, center, 1)
    assert one["center"] == center and {n["depth"] for n in one["nodes"]} == {0, 1}
    types1 = {n["type"] for n in one["nodes"]}
    assert {"concept", "question", "note"} <= types1
    ids = {n["id"] for n in one["nodes"]}
    assert all(e["from"] in ids and e["to"] in ids for e in one["edges"])
    two = subgraph(conn, notebook.id, center, 2)
    assert (
        len(two["nodes"]) > len(one["nodes"])
        and max(n["depth"] for n in two["nodes"]) == 2
    )
    rels = {e["relation"] for e in two["edges"]}
    assert {"mentions", "note_mentions", "answered_by"} <= rels
    assert subgraph(conn, notebook.id, center, 9)["depth"] == 2  # clamped


def test_subgraph_filters_and_recentre(conn, notebook, data):
    center = node_id(conn, f"concept:{TERM}")
    only_notes = subgraph(conn, notebook.id, center, 2, {"note"})
    assert {n["type"] for n in only_notes["nodes"] if n["id"] != center} == {"note"}
    note_node = node_id(conn, f"note:{data['note'].id}")
    re_centered = subgraph(conn, notebook.id, note_node, 1)
    assert re_centered["center"] == note_node
    assert {n["type"] for n in re_centered["nodes"]} >= {"note", "source_evidence"}


def test_subgraph_bounds_on_a_dense_graph(conn, notebook):
    for i in range(80):
        ex = start_exchange(
            conn, notebook.id, None, f"Question {i} about spectral theory?"
        )
        complete_message(
            conn, ex.assistant_message_id, f"Answer {i} on spectral theory.", model="q"
        )
        sync_turn(conn, ex.assistant_message_id)
    center = node_id(conn, "concept:spectral theory")
    g = subgraph(conn, notebook.id, center, 2)
    assert (
        len(g["nodes"]) <= MAX_NODES and len(g["edges"]) <= MAX_EDGES and g["truncated"]
    )
    payload = json.dumps(g)
    assert len(payload) < 20000


def test_graph_isolation_and_inspectors(conn, notebook, data, settings, store):
    other = create_notebook(conn, "other")
    center = node_id(conn, f"concept:{TERM}")
    assert subgraph(conn, other.id, center, 1) is None
    assert subgraph(conn, notebook.id, 987654, 1) is None
    edge = subgraph(conn, notebook.id, center, 1)["edges"][0]
    detail = edge_detail(conn, notebook.id, edge["id"])
    assert detail["relation"] == edge["relation"] and detail["provenance"]
    assert edge_detail(conn, other.id, edge["id"]) is None
    assert search_nodes(conn, notebook.id, TERM)[0]["type"] == "concept"
    assert search_nodes(conn, other.id, TERM) == []
    assert search_nodes(conn, notebook.id, '"((NEAR') == []
    assert default_center(conn, notebook.id) is not None
    assert default_center(conn, other.id) is None
    # deleted source: the evidence node (snapshot) is still inspectable
    delete_source(conn, settings, notebook.id, data["chunk"][1], store)
    ev = node_id(conn, "evidence:1")
    assert subgraph(conn, notebook.id, ev, 1) is not None


def test_correction_nodes_in_graph(conn, notebook, data):
    set_outcome(
        conn,
        notebook.id,
        data["ex"].assistant_message_id,
        "corrected",
        f"A {TERM} actually reflects sunlight.",
    )
    corr = node_id(conn, f"correction:{data['ex'].assistant_message_id}")
    g = subgraph(conn, notebook.id, corr, 1)
    assert "corrects" in {e["relation"] for e in g["edges"]}


# --- web ----------------------------------------------------------------------------


@pytest.fixture
def web(tmp_path):
    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    conn = open_db(s.db_path)
    nb = create_notebook(conn, "Research")
    d = build(conn, s, nb, tmp_path)
    other = create_notebook(conn, "Other")
    conn.close()
    holder = {}

    def factory():
        holder["w"] = WorkerEmbedder(
            idle_seconds=60,
            env={
                "LITENOTEBOOK_EMBED_WORKER_BACKEND": "test-fake",
                "LITENOTEBOOK_EMBED_WORKER_TEST_DIR": str(tmp_path),
            },
        )
        return holder["w"]

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
            "nb": nb.id,
            "other": other.id,
            "d": d,
            "tmp": tmp_path,
            "holder": holder,
        }


def test_search_page_lexical_does_not_spawn_worker(web):
    c, nb = web["c"], web["nb"]
    page = c.get(f"/notebooks/{nb}/search")
    assert page.status_code == 200 and "Fast keyword" in page.text
    r = c.get(f"/notebooks/{nb}/search?q={TERM}", headers=HX_SEARCH)
    assert r.text.strip().startswith('<div id="search-results"')
    sections = dict(
        re.findall(r'id="sec-(\w+)">\w+ <span class="muted">\((\d+)', r.text)
    )
    assert len(sections) == 5 and sections.pop("annotations") == "0"
    assert all(int(v) > 0 for v in sections.values())
    for mode in ("sources", "notes", "chats", "memory"):
        assert c.get(f"/notebooks/{nb}/search?q={TERM}&mode={mode}").status_code == 200
    assert pids(web["tmp"]) == [] and web["holder"] == {}
    huge = c.get(f"/notebooks/{nb}/search", params={"q": "(" * 3000 + '"'})
    assert huge.status_code == 200
    html = c.get(f"/notebooks/{nb}/search", params={"q": "<script>alert(1)</script>"})
    assert "<script>alert(1)" not in html.text


def test_hybrid_search_spawns_the_worker(web):
    c, nb = web["c"], web["nb"]
    r = c.get(f"/notebooks/{nb}/search?q={TERM}&mode=sources&engine=hybrid")
    assert r.status_code == 200 and "hybrid source search" in r.text
    assert "embedding model was loaded for this search" in r.text
    assert len(pids(web["tmp"])) == 1
    pid = web["holder"]["w"].pid
    assert "semantic" in r.text
    c.get(f"/notebooks/{nb}/search?q={TERM}&mode=sources&engine=hybrid")
    assert web["holder"]["w"].pid == pid  # warm reuse
    assert c.get("/healthz").json()["embedding_worker"]["status"] == "ready"


def test_search_results_are_scoped_on_the_web(web):
    c = web["c"]
    r = c.get(f"/notebooks/{web['other']}/search?q={TERM}", headers=HX_SEARCH).text
    assert re.findall(r"\((\d+)\)", r) == ["0", "0", "0", "0", "0"]


def test_graph_routes(web):
    c, nb = web["c"], web["nb"]
    page = c.get(f"/notebooks/{nb}/memory/graph")
    center = int(re.search(r'data-center="(\d+)"', page.text).group(1))
    assert '<script src="/static/graph.js" defer></script>' in page.text
    j = c.get(f"/notebooks/{nb}/memory/graph.json?node={center}&depth=2").json()
    assert j["nodes"] and j["limits"] == {"nodes": 50, "edges": 100, "depth": 2}
    filt = c.get(
        f"/notebooks/{nb}/memory/graph.json?node={center}&depth=2&types=note"
    ).json()
    assert {n["type"] for n in filt["nodes"] if n["id"] != center} <= {"note"}
    bogus = c.get(f"/notebooks/{nb}/memory/graph.json?node={center}&types=evil;drop")
    assert bogus.status_code == 200  # unknown types ignored
    found = c.get(f"/notebooks/{nb}/memory/graph/search?q={TERM}").json()["results"]
    assert found and found[0]["label"] == TERM
    edge = j["edges"][0]["id"]
    assert (
        c.get(
            f"/notebooks/{nb}/memory/edges/{edge}", headers={"HX-Request": "true"}
        ).status_code
        == 200
    )
    node = c.get(
        f"/notebooks/{nb}/memory/nodes/{center}", headers={"HX-Request": "true"}
    )
    assert "Open in graph" in node.text and "Add to note" in node.text
    other = web["other"]
    for url in (
        f"/notebooks/{other}/memory/graph.json?node={center}",
        f"/notebooks/{other}/memory/edges/{edge}",
        f"/notebooks/{other}/memory/nodes/{center}",
        f"/notebooks/{other}/memory/graph?node={center}",
        f"/notebooks/{nb}/memory/graph.json?node=99999",
    ):
        assert c.get(url).status_code == 404, url
    assert c.get(f"/notebooks/{other}/memory/graph/search?q={TERM}").json() == {
        "results": []
    }
    assert c.get(f"/notebooks/{other}/memory/graph").status_code == 200  # empty state
    assert pids(web["tmp"]) == []


def test_graph_labels_are_escaped_and_json_is_inert(web):
    c, nb = web["c"], web["nb"]
    from litenotebook.db import connect

    conn = connect(Path(web["tmp"]) / "data" / "litenotebook.sqlite3")
    note = create_note(conn, nb, "<svg onload=alert(1)> heliostat", "x")
    conn.close()
    j = c.get(f"/notebooks/{nb}/memory/graph/search?q=heliostat")
    assert j.headers["content-type"].startswith("application/json")
    html = c.get(
        f"/notebooks/{nb}/memory/nodes/"
        + str(next(r["id"] for r in j.json()["results"] if "svg" in r["label"])),
        headers={"HX-Request": "true"},
    ).text
    assert "<svg onload" not in html and "&lt;svg onload" in html
    js = c.get("/static/graph.js").text
    assert ".innerHTML" not in js and "insertAdjacentHTML" not in js
    assert note.id


def test_phase7_pages_never_import_the_ml_stack(tmp_path):
    code = f"""
import sys, httpx
sys.path.insert(0, {str(Path(__file__).parent)!r})
from fastapi.testclient import TestClient
from conftest import FakeEmbedder
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.notebooks import create_notebook
from litenotebook.notes import create_note
from litenotebook.chat import start_exchange, complete_message
from litenotebook.memory.graph import sync_turn
from litenotebook.web.app import create_app
s = load_settings({{"LITENOTEBOOK_DATA_DIR": {str(tmp_path / "d")!r}}})
c = open_db(s.db_path)
nb = create_notebook(c, "n")
ex = start_exchange(c, nb.id, None, "What is a heliostat?")
complete_message(c, ex.assistant_message_id, "A mirror.", model="q")
sync_turn(c, ex.assistant_message_id)
create_note(c, nb.id, "Heliostat note", "mirrors")
c.close()
calls = []
def factory():
    calls.append(1)
    raise AssertionError("must not create the embedder")
def llama(r):
    return httpx.Response(200, json={{"data": []}})
app = create_app(s, env={{}}, embedder_factory=factory,
                 llm_transport=httpx.MockTransport(llama))
h = {{"Origin": "{BASE}"}}
with TestClient(app, base_url="{BASE}") as cl:
    base = "/notebooks/1"
    for url in ("/notes", "/notes/1", "/search", "/search?q=heliostat",
                "/search?q=heliostat&mode=notes", "/memory/graph",
                "/memory/graph/search?q=heliostat", "/notes/1/attach?q=mirror"):
        assert cl.get(base + url).status_code == 200, url
    found = cl.get(base + "/memory/graph/search?q=heliostat").json()
    node = found["results"][0]["id"]
    graph = cl.get(f"{{base}}/memory/graph.json?node={{node}}&depth=2")
    assert graph.status_code == 200
    form = {{"title": "Heliostat note", "content": "edited", "revision": "1"}}
    assert cl.post(base + "/notes/1/autosave", headers=h, data=form).status_code == 200
heavy = ("torch", "sentence_transformers", "transformers", "faiss")
print("HEAVY", [m for m in heavy if m in sys.modules], len(calls))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "HEAVY [] 0" in out.stdout
    assert process_gone  # imported for symmetry with worker tests
