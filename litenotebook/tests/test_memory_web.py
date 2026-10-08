"""Phase 5 web: memory in chat ([M#]), feedback, Memory page, CLI, security."""

import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook import cli
from litenotebook.chat import (
    complete_message,
    get_message,
    list_messages,
    start_exchange,
)
from litenotebook.config import load_settings
from litenotebook.db import connect, open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import delete_source, ingest_pdf
from litenotebook.memory.graph import check, get_outcome, sync_turn
from litenotebook.notebooks import create_notebook
from litenotebook.services.chat import NO_EVIDENCE_OR_MEMORY, NO_READY_SOURCES
from litenotebook.services.settings import SettingsService
from litenotebook.web.app import create_app
from test_chat_web import ORIGIN, FakeLlama, ask, parse_events

BASE = "http://127.0.0.1:5056"
HX = {**ORIGIN, "HX-Request": "true"}
SECRET = "sk-memory-SECRET-999"


class ScriptedLlama(FakeLlama):
    """Answers with fixed text; records every generation request."""

    def __init__(self, text="Noted."):
        super().__init__(chunks=[text])

    @property
    def calls(self):
        return len(self.requests)


@pytest.fixture
def app_settings(tmp_path):
    return load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})


@pytest.fixture
def llama():
    return ScriptedLlama()


def make_app(app_settings, llama):
    return create_app(
        app_settings,
        env={},
        llm_transport=httpx.MockTransport(llama),
        embedder_factory=FakeEmbedder,
    )


@pytest.fixture
def client(app_settings, llama):
    with TestClient(make_app(app_settings, llama), base_url=BASE) as c:
        yield c


def db(app_settings):
    return connect(app_settings.db_path)


def remember(app_settings, notebook_id, question, answer, *, cid=None, model="qwen"):
    """A completed past exchange, synced into the graph (no LLM)."""
    conn = db(app_settings)
    try:
        ex = start_exchange(conn, notebook_id, cid, question)
        complete_message(conn, ex.assistant_message_id, answer, model=model)
        sync_turn(conn, ex.assistant_message_id)
        return ex
    finally:
        conn.close()


@pytest.fixture
def nb(app_settings):
    conn = open_db(app_settings.db_path)
    try:
        return create_notebook(conn, "Research").id
    finally:
        conn.close()


def add_source(app_settings, notebook_id, tmp_path, pages):
    conn = db(app_settings)
    try:
        from litenotebook.notebooks import get_notebook

        pdf = make_pdf(tmp_path / f"src{len(pages)}.pdf", pages)
        ingest_pdf(
            conn,
            app_settings,
            get_notebook(conn, notebook_id),
            pdf,
            FakeEmbedder(),
            FaissStore(conn, app_settings),
        )
        return conn.execute("SELECT max(id) FROM sources").fetchone()[0]
    finally:
        conn.close()


def last_user_prompt(llama):
    return llama.requests[-1]["messages"][-1]["content"]


# --- memory in chat --------------------------------------------------------------


def test_memory_only_question_uses_memory_and_m_citations(
    client, app_settings, nb, llama
):
    remember(
        app_settings,
        nb,
        "What did we call the calibration system?",
        "The calibration system was called Helios.",
    )
    llama.chunks = ["It was Helios [M1]."]
    r, events = ask(client, nb, "What calibration system did we discuss earlier?")
    assert llama.calls == 1 and events[-1][0] == "done"
    status = [d for e, d in events if e == "status"][-1]
    assert status == {"stage": "generating", "sources": 0, "memories": 1}
    prompt = last_user_prompt(llama)
    assert (
        "=== RELEVANT MEMORY" in prompt and "(no matching document passages)" in prompt
    )
    assert "Earlier answer: The calibration system was called Helios." in prompt
    conn = db(app_settings)
    msg = get_message(conn, nb, events[0][1]["message_id"])
    conn.close()
    assert [r.label for r in msg.memory_refs] == [1] and msg.citations == []
    ref = msg.memory_refs[0]
    assert ref.question == "What did we call the calibration system?"
    assert ref.excerpt == "The calibration system was called Helios."
    assert ref.trust == "Uncited assistant answer"


def test_source_and_memory_combined_in_one_call(
    client, app_settings, nb, llama, tmp_path
):
    add_source(
        app_settings, nb, tmp_path, ["The Borealis telescope aperture is 3.7 m."]
    )
    remember(app_settings, nb, "What is the telescope aperture?", "About 3.7 metres.")
    llama.chunks = ["It is 3.7 m [S1], as we found before [M1]."]
    _, events = ask(client, nb, "What is the telescope aperture?")
    assert llama.calls == 1
    messages = llama.requests[-1]["messages"]
    user = messages[-1]["content"]
    order = [user.index(m) for m in ("=== RELEVANT MEMORY", "=== SOURCES", "QUESTION:")]
    assert order == sorted(order)  # memory, then sources, then question
    assert messages[0]["role"] == "system" and "[M1]" in messages[0]["content"]
    conn = db(app_settings)
    msg = get_message(conn, nb, events[0][1]["message_id"])
    conn.close()
    assert [c.label for c in msg.citations] == [1] and [
        r.label for r in msg.memory_refs
    ] == [1]


def test_source_only_question_has_no_memory_block(
    client, app_settings, nb, llama, tmp_path
):
    add_source(app_settings, nb, tmp_path, ["Quaternions extend complex numbers."])
    llama.chunks = ["They extend complex numbers [S1]."]
    ask(client, nb, "What do quaternions extend?")
    assert "RELEVANT MEMORY" not in last_user_prompt(llama)


def test_no_evidence_and_no_memory_makes_no_llm_call(
    client, app_settings, nb, llama, tmp_path, monkeypatch
):
    _, events = ask(client, nb, "What is the airspeed of a swallow?")
    conn = db(app_settings)
    assert get_message(conn, nb, events[0][1]["message_id"]).content == NO_READY_SOURCES
    conn.close()
    add_source(app_settings, nb, tmp_path, ["Completely unrelated text about ducks."])
    remember(app_settings, nb, "Ducks?", "Ducks quack.")
    # FAISS always returns nearest chunks; simulate retrieval finding nothing
    from litenotebook.retrieval.hybrid import Retriever

    monkeypatch.setattr(Retriever, "search", lambda self, *a, **k: [])
    _, events = ask(client, nb, "zzzz qqqq xxxx")  # and no memory matches either
    conn = db(app_settings)
    msg = get_message(conn, nb, events[0][1]["message_id"])
    conn.close()
    assert msg.content == NO_EVIDENCE_OR_MEMORY and msg.model is None
    assert llama.calls == 0


@pytest.mark.parametrize(
    ("answer", "valid", "rendered"),
    [
        ("A [M1].", [1], 1),
        ("A [M1][M2].", [1, 2], 2),
        ("A [M1, M2] and again [M1].", [1, 2], 3),
        ("A [M99] and [M2].", [2], 1),
    ],
)
def test_memory_reference_forms(
    client, app_settings, nb, llama, answer, valid, rendered
):
    remember(app_settings, nb, "Helios calibration?", "Helios calibrates.")
    remember(app_settings, nb, "Helios service date?", "Helios started in 2011.")
    llama.chunks = [answer]
    _, events = ask(client, nb, "Tell me about Helios")
    mid = events[0][1]["message_id"]
    conn = db(app_settings)
    msg = get_message(conn, nb, mid)
    conn.close()
    assert [r.label for r in msg.memory_refs] == valid
    if "M99" in answer:
        assert msg.meta["invalid_memory_labels"] == [99]
    html = client.get(f"/notebooks/{nb}/chat/messages/{mid}", headers=HX).text
    body = html.split('class="msg-body">', 1)[1].split("</div>", 1)[0]
    assert len(re.findall(r'class="cite cite-memory"', body)) == rendered
    assert ">M99<" not in body


def test_memory_evidence_survives_restart_rename_and_source_deletion(
    app_settings, nb, llama, tmp_path
):
    sid = add_source(app_settings, nb, tmp_path, ["The aperture is 3.7 metres."])
    with TestClient(make_app(app_settings, llama), base_url=BASE) as c:
        llama.chunks = ["3.7 metres [S1]."]
        _, first = ask(c, nb, "What is the aperture?")
        llama.chunks = ["As before, 3.7 m [M1]."]
        _, second = ask(c, nb, "Remind me of the aperture we discussed")
    mid = second[0][1]["message_id"]
    conn = db(app_settings)
    [ref] = get_message(conn, nb, mid).memory_refs
    delete_source(conn, app_settings, nb, sid, FaissStore(conn, app_settings))
    conn.close()
    with TestClient(make_app(app_settings, llama), base_url=BASE) as c:  # "restart"
        c.post(
            f"/notebooks/{nb}/chat/{first[0][1]['conversation_id']}/rename",
            data={"title": "Renamed later"},
            headers=HX,
        )
        panel = c.get(f"/notebooks/{nb}/memory/refs/{ref.id}", headers=HX).text
        page = c.get(f"/notebooks/{nb}/memory/refs/{ref.id}")
    assert "Original source has been deleted." in panel
    assert "What is the aperture?" in panel and "3.7 metres" in panel
    assert "Source-backed</dt><dd>yes" in panel
    assert "What is the aperture?" in panel  # snapshot title kept, not the rename
    assert page.status_code == 200


# --- feedback ------------------------------------------------------------------------


def test_feedback_controls_update_memory_without_llm(client, app_settings, nb, llama):
    ex = remember(
        app_settings,
        nb,
        "When did Helios enter service?",
        "Helios entered service in 2010.",
    )
    base = f"/notebooks/{nb}/chat/messages/{ex.assistant_message_id}"
    html = client.get(base, headers=HX).text
    assert (
        'value="useful"' in html and 'value="dead_end"' in html and "Correct…" in html
    )
    r = client.post(f"{base}/outcome", data={"outcome": "useful"}, headers=HX)
    assert r.status_code == 200 and 'aria-pressed="true">Useful' in r.text
    form = client.get(f"{base}/correct", headers=HX).text
    assert 'name="correction"' in form
    bad = client.post(
        f"{base}/outcome", data={"outcome": "corrected", "correction": " "}, headers=HX
    )
    assert bad.status_code == 422 and "Enter the correction" in bad.text
    fix = client.post(
        f"{base}/outcome",
        data={
            "outcome": "corrected",
            "correction": "Helios entered service in 2011. <script>alert(1)</script>",
        },
        headers=HX,
    )
    assert fix.status_code == 200 and "<script>alert(1)" not in fix.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in fix.text
    assert "Helios entered service in 2010." in fix.text  # original kept
    assert (
        client.post(
            f"{base}/outcome", data={"outcome": "brilliant"}, headers=HX
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"{base}/outcome", data={"outcome": "dead_end"}, headers=HX
        ).status_code
        == 200
    )
    conn = db(app_settings)
    assert get_outcome(conn, ex.assistant_message_id) == ("dead_end", None)
    assert check(conn, nb).ok
    conn.close()
    assert llama.calls == 0


def test_retroactive_correction_changes_future_memory_context(
    client, app_settings, nb, llama
):
    old = remember(
        app_settings,
        nb,
        "When did Helios enter service?",
        "Helios entered service in 2010.",
    )
    client.post(
        f"/notebooks/{nb}/chat/messages/{old.assistant_message_id}/outcome",
        data={"outcome": "corrected", "correction": "Helios entered service in 2011."},
        headers=HX,
    )
    llama.chunks = ["2011 [M1]."]
    ask(client, nb, "When did Helios enter service?")
    prompt = last_user_prompt(llama)
    m1 = prompt.split("[M1]", 1)[1].split("[M2]", 1)[0]
    assert "User correction" in m1 and "2011" in m1
    assert "superseded" in prompt.split("[M2]", 1)[1]


def test_dead_end_memory_is_left_out_of_chat(client, app_settings, nb, llama):
    ex = remember(app_settings, nb, "What is the Zorbanian constant?", "It is 42.")
    client.post(
        f"/notebooks/{nb}/chat/messages/{ex.assistant_message_id}/outcome",
        data={"outcome": "dead_end"},
        headers=HX,
    )
    _, events = ask(client, nb, "Zorbanian constant?")
    assert llama.calls == 0  # nothing usable: no sources, dead-end memory only
    conn = db(app_settings)
    assert get_message(conn, nb, events[0][1]["message_id"]).content == NO_READY_SOURCES
    conn.close()


# --- Memory page ---------------------------------------------------------------------


def test_memory_page_stats_filters_search_and_inspector(client, app_settings, nb):
    a = remember(app_settings, nb, "What is the telescope aperture?", "3.7 metres.")
    b = remember(app_settings, nb, "When did Helios enter service?", "In 2010.")
    for mid, data in (
        (a.assistant_message_id, {"outcome": "useful"}),
        (b.assistant_message_id, {"outcome": "corrected", "correction": "In 2011."}),
    ):
        client.post(
            f"/notebooks/{nb}/chat/messages/{mid}/outcome", data=data, headers=HX
        )
    page = client.get(f"/notebooks/{nb}/memory")
    assert page.status_code == 200
    assert re.search(r"<dt>Useful</dt><dd>1</dd>", page.text)
    assert re.search(r"<dt>Corrected</dt><dd>1</dd>", page.text)
    assert "telescope aperture" in page.text
    useful = client.get(f"/notebooks/{nb}/memory?filter=useful").text
    assert "telescope aperture" in useful and "Helios" not in useful
    corrections = client.get(f"/notebooks/{nb}/memory?filter=corrections").text
    assert "In 2011." in corrections
    assert "What is the telescope aperture?" not in corrections
    found = client.get(
        f"/notebooks/{nb}/memory?q=telescope+aperture",
        headers={"HX-Request": "true", "HX-Target": "memory-results"},
    )
    assert found.text.startswith('<div id="memory-results">') and "score" in found.text
    node = re.search(r"/memory/nodes/(\d+)", found.text).group(1)
    detail = client.get(f"/notebooks/{nb}/memory/nodes/{node}")
    assert detail.status_code == 200 and "Connections" in detail.text
    assert "answered_by" in detail.text
    assert (
        client.get(f"/notebooks/{nb}?tab=memory", follow_redirects=False).headers[
            "location"
        ]
        == f"/notebooks/{nb}/memory"
    )


def test_memory_content_is_escaped(client, app_settings, nb):
    remember(
        app_settings,
        nb,
        "<img src=x onerror=alert(1)> question?",
        "<script>alert('answer')</script> about widgets",
    )
    for url in (f"/notebooks/{nb}/memory", f"/notebooks/{nb}/memory?q=widgets"):
        html = client.get(url).text
        assert "<script>alert(" not in html and "<img src=x" not in html
        assert "&lt;script&gt;" in html


def test_cross_notebook_memory_access_denied(client, app_settings, nb, llama):
    remember(app_settings, nb, "Helios calibration?", "Helios calibrates.")
    llama.chunks = ["Helios [M1]."]
    _, events = ask(client, nb, "Helios calibration?")
    mid = events[0][1]["message_id"]
    conn = db(app_settings)
    [ref] = get_message(conn, nb, mid).memory_refs
    node = conn.execute(
        "SELECT id FROM memory_nodes WHERE notebook_id = ? LIMIT 1", (nb,)
    ).fetchone()[0]
    other = create_notebook(conn, "Other").id
    conn.close()
    for method, url, data in [
        ("GET", f"/notebooks/{other}/memory/nodes/{node}", None),
        ("GET", f"/notebooks/{other}/memory/refs/{ref.id}", None),
        (
            "POST",
            f"/notebooks/{other}/chat/messages/{mid}/outcome",
            {"outcome": "useful"},
        ),
        ("GET", f"/notebooks/{other}/chat/messages/{mid}/correct", None),
    ]:
        assert client.request(method, url, data=data, headers=HX).status_code == 404
    other_page = client.get(f"/notebooks/{other}/memory?q=Helios").text
    assert "Helios calibrates" not in other_page
    assert client.get("/notebooks/424242/memory").status_code == 404


def test_feedback_requires_same_origin(client, app_settings, nb):
    ex = remember(app_settings, nb, "Q?", "A.")
    url = f"/notebooks/{nb}/chat/messages/{ex.assistant_message_id}/outcome"
    r = client.post(url, data={"outcome": "useful"}, headers={"Origin": "http://evil"})
    assert r.status_code == 403
    conn = db(app_settings)
    assert get_outcome(conn, ex.assistant_message_id) is None
    conn.close()


def test_api_key_never_on_memory_pages(app_settings, nb, llama):
    conn = db(app_settings)
    SettingsService(conn, env={}).update(llm_api_key=SECRET)
    conn.close()
    remember(app_settings, nb, "Helios?", "Helios calibrates.")
    with TestClient(make_app(app_settings, llama), base_url=BASE) as c:
        llama.chunks = ["Helios [M1]."]
        r, events = ask(c, nb, "Helios?")
        pages = [r.text] + [
            c.get(u).text
            for u in (
                f"/notebooks/{nb}/memory",
                f"/notebooks/{nb}/memory?q=helios",
                f"/notebooks/{nb}/chat",
            )
        ]
    assert llama.headers[-1]["authorization"] == f"Bearer {SECRET}"
    assert all(SECRET not in p for p in pages)


def test_startup_backfills_existing_conversations(app_settings, nb, llama):
    conn = db(app_settings)
    ex = start_exchange(conn, nb, None, "Pre-Phase-5 question about eigenvectors?")
    complete_message(conn, ex.assistant_message_id, "Old answer.", model="qwen")
    conn.close()  # no sync: as if written before graph memory existed
    with TestClient(make_app(app_settings, llama), base_url=BASE) as c:
        page = c.get(f"/notebooks/{nb}/memory?q=eigenvectors").text
    assert "Pre-Phase-5 question about eigenvectors?" in page
    with TestClient(make_app(app_settings, llama), base_url=BASE):
        pass
    conn = db(app_settings)
    report = check(conn, nb)
    conn.close()
    assert report.ok and report.synced_exchanges == 1 and llama.calls == 0


# --- CLI -----------------------------------------------------------------------------


def test_memory_cli_rebuild_check_search(app_settings, nb, capsys):
    a = remember(app_settings, nb, "What is the telescope aperture?", "3.7 metres.")
    remember(app_settings, nb, "Calibration system name?", "Helios.")
    conn = db(app_settings)
    from litenotebook.memory.graph import set_outcome

    set_outcome(conn, nb, a.assistant_message_id, "useful")
    conn.close()
    base = ["--data-dir", str(app_settings.data_dir)]
    assert cli.main([*base, "memory", "check", "--notebook", "Research"]) == 0
    assert "Memory OK" in capsys.readouterr().out
    assert cli.main([*base, "memory", "rebuild", "--notebook", str(nb)]) == 0
    assert "outcomes and corrections kept: 1" in capsys.readouterr().out
    assert (
        cli.main(
            [*base, "memory", "search", "--notebook", "Research", "telescope aperture"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out.startswith("#1  exchange") and "useful" in out and "ms" in out
    assert cli.main([*base, "memory", "search", "--notebook", "Research", "zzzz"]) == 0
    assert "No memory matches." in capsys.readouterr().out
    assert cli.main([*base, "memory", "check", "--notebook", "missing"]) == 1


# --- lazy loading --------------------------------------------------------------------


def test_memory_pages_and_history_do_not_load_models(tmp_path):
    code = f"""
import sys, httpx
sys.path.insert(0, {str(Path(__file__).parent)!r})
from fastapi.testclient import TestClient
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.notebooks import create_notebook
from litenotebook.chat import start_exchange, complete_message
from litenotebook.memory.graph import set_outcome
from litenotebook.web.app import create_app
s = load_settings({{"LITENOTEBOOK_DATA_DIR": {str(tmp_path / "d")!r}}})
c = open_db(s.db_path)
nb = create_notebook(c, "n")
ex = start_exchange(c, nb.id, None, "What is the telescope aperture?")
complete_message(c, ex.assistant_message_id, "3.7 metres.", model="q")
c.close()
calls = []
def factory():
    calls.append(1)
    raise AssertionError("MiniLM must not load")
def llama(r):
    return httpx.Response(200, json={{"data": []}})
app = create_app(s, env={{}}, embedder_factory=factory,
                 llm_transport=httpx.MockTransport(llama))
h = {{"Origin": "{BASE}", "HX-Request": "true"}}
with TestClient(app, base_url="{BASE}") as cl:   # startup backfill runs here
    for url in ("/", "/notebooks/1", "/notebooks/1/chat/1", "/notebooks/1/memory",
                "/notebooks/1/memory?q=telescope", "/notebooks/1/memory?filter=useful",
                "/notebooks/1/memory/nodes/1"):
        assert cl.get(url).status_code == 200, url
    assert cl.post("/notebooks/1/chat/messages/2/outcome", data={{"outcome": "useful"}},
                   headers=h).status_code == 200
heavy = ("torch", "sentence_transformers", "transformers", "faiss")
print("HEAVY", [m for m in heavy if m in sys.modules], len(calls))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "HEAVY [] 0" in out.stdout


def test_chat_history_messages_carry_memory_and_outcomes(app_settings, nb):
    ex = remember(app_settings, nb, "Q about Helios?", "Helios.")
    conn = db(app_settings)
    from litenotebook.memory.graph import set_outcome

    set_outcome(conn, nb, ex.assistant_message_id, "corrected", "Helios-2.")
    [_, answer] = list_messages(conn, ex.conversation.id)
    conn.close()
    assert answer.outcome == "corrected" and answer.correction == "Helios-2."
    assert answer.rateable and json.dumps(answer.meta) is not None
    assert parse_events("event: x\ndata: {}\n\n") == [("x", {})]
