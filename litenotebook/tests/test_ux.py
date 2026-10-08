"""Phase 9: shell, command palette, dialogs, feedback, accessibility, errors."""

import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.chat import CitedChunk, complete_message, end_message, start_exchange
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.notebooks import create_notebook
from litenotebook.notes import create_note, link_chunk
from litenotebook.reader.annotations import create_annotation
from litenotebook.web.app import create_app
from litenotebook.web.navigation import FLASH_COOKIE, FLASH_MESSAGES

BASE = "http://127.0.0.1:5056"
ORIGIN = {"Origin": BASE}
HX = {**ORIGIN, "HX-Request": "true"}
PAGE = {"Accept": "text/html,application/xhtml+xml"}
STATIC = Path(__file__).parents[1] / "src/litenotebook/web/static"
EVIL = '<script>alert("x")</script>'


class Audit(HTMLParser):
    """Collects ids, headings, form fields, labels and accessible names."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.headings: list[str] = []
        self.fields: list[dict] = []
        self.label_for: set[str] = set()
        self.label_depth = 0
        self.controls: list[dict] = []
        self.open_control: dict | None = None
        self.dialogs: list[dict] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(tag)
        if tag == "label":
            self.label_depth += 1
            if a.get("for"):
                self.label_for.add(a["for"])
        if tag in ("input", "textarea", "select") and a.get("type") != "hidden":
            self.fields.append({**a, "_in_label": self.label_depth > 0})
        if tag in ("button", "a") and (tag == "button" or "href" in a):
            self.open_control = {**a, "_text": ""}
            self.controls.append(self.open_control)
        if tag == "dialog":
            self.dialogs.append(a)

    def handle_endtag(self, tag):
        if tag == "label":
            self.label_depth -= 1
        if tag in ("button", "a"):
            self.open_control = None

    def handle_data(self, data):
        if self.open_control is not None:
            self.open_control["_text"] += data


def audit(html: str) -> Audit:
    a = Audit()
    a.feed(html)
    return a


def assert_accessible(html: str, where: str) -> None:
    a = audit(html)
    dupes = {i for i in a.ids if a.ids.count(i) > 1}
    assert not dupes, f"{where}: duplicate ids {dupes}"
    assert a.headings.count("h1") == 1, f"{where}: h1 count {a.headings}"
    assert a.headings[0] == "h1", f"{where}: first heading is {a.headings[0]}"
    for f in a.fields:
        labelled = (
            f.get("id") in a.label_for
            or f["_in_label"]
            or f.get("aria-label")
            or f.get("aria-labelledby")
        )
        assert labelled, f"{where}: unlabelled field {f}"
    for c in a.controls:
        name = (c.get("aria-label") or c["_text"] or c.get("title") or "").strip()
        assert name, f"{where}: control without a name {c}"
    for d in a.dialogs:
        target = d.get("aria-labelledby")
        assert target and target in a.ids, f"{where}: unlabelled dialog {d}"
    assert 'class="skip-link" href="#main"' in html and 'id="main"' in html
    assert '<nav class="nav" aria-label="Main">' in html


@pytest.fixture
def web(tmp_path):
    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    conn = open_db(s.db_path)
    nb = create_notebook(conn, "Research", "Solar notes")
    other = create_notebook(conn, "Other " + EVIL)
    pages = [
        "Intro to heliostats.",
        "The Borealis heliostat is precise to 0.1 mrad.",
        "End.",
    ]
    ingest_pdf(
        conn,
        s,
        nb,
        make_pdf(tmp_path / "solar.pdf", pages),
        FakeEmbedder(),
        FaissStore(conn, s),
    )
    chunk = conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks WHERE page_no = 2"
    ).fetchone()
    ex = start_exchange(conn, nb.id, None, "How precise is it?")
    complete_message(
        conn,
        ex.assistant_message_id,
        "Very precise [S1]. " + EVIL,
        model="q",
        citations=[
            CitedChunk(1, chunk[0], chunk[1], "solar.pdf", chunk[2], chunk[3], chunk[4])
        ],
    )
    failed = start_exchange(conn, nb.id, ex.conversation.id, "And the receiver?")
    end_message(conn, failed.assistant_message_id, "failed", "LLM unavailable")
    note = create_note(conn, nb.id, "Reading list", "heliostat accuracy")
    link_chunk(conn, nb.id, note.id, chunk[0])
    ann = create_annotation(
        conn,
        s,
        nb.id,
        1,
        page_no=2,
        selected_text="Borealis",
        comment=EVIL,
        rects=[[0.1, 0.1, 0.3, 0.12]],
    )
    conn.close()
    calls: list[int] = []

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
            "conv": ex.conversation.id,
            "answer": ex.assistant_message_id,
            "failed": failed.assistant_message_id,
            "note": note.id,
            "ann": ann.id,
            "chunk": chunk[0],
            "calls": calls,
        }


def pages(w):
    nb, conv, note = w["nb"], w["conv"], w["note"]
    return {
        "index": "/",
        "global-search": "/search?q=heliostat",
        "settings": "/settings",
        "sources": f"/notebooks/{nb}",
        "chat": f"/notebooks/{nb}/chat/{conv}",
        "notes": f"/notebooks/{nb}/notes/{note}",
        "memory": f"/notebooks/{nb}/memory",
        "graph": f"/notebooks/{nb}/memory/graph",
        "search": f"/notebooks/{nb}/search?q=heliostat",
        "source": f"/notebooks/{nb}/sources/1",
        "reader": f"/notebooks/{nb}/sources/1/read",
    }


# --- command palette ----------------------------------------------------------------


def test_palette_markup_in_every_page(web):
    html = web["c"].get("/").text
    assert (
        '<dialog class="dialog palette" id="palette" aria-labelledby="palette-title">'
        in html
    )
    assert 'role="combobox"' in html and 'role="listbox"' in html
    assert 'data-palette-open aria-keyshortcuts="Control+K"' in html
    assert '<script src="/static/app.js" defer></script>' in html


def test_palette_commands_global_and_notebook_scoped(web):
    c, nb, other = web["c"], web["nb"], web["other"]
    plain = c.get("/palette.json").json()["commands"]
    labels = [x["label"] for x in plain]
    assert labels[:5] == [
        "Notebooks",
        "New notebook",
        "Search all notebooks",
        "Settings",
        "Keyboard shortcuts",
    ]
    assert (
        "Open notebook: Research" in labels and "Open notebook: Other " + EVIL in labels
    )
    scoped = c.get(f"/palette.json?nb={nb}").json()["commands"]
    assert scoped[0]["group"] == "Research"  # notebook actions rank first
    urls = {x["label"]: x["url"] for x in scoped}
    assert urls["New chat"] == f"/notebooks/{nb}/chat/new"
    assert urls["New note"] == f"/notebooks/{nb}/notes?new=1"
    assert urls["Memory graph"] == f"/notebooks/{nb}/memory/graph"
    assert urls["Read: solar.pdf"] == f"/notebooks/{nb}/sources/1/read"
    assert "Open notebook: Research" not in urls  # it is the current one
    current = [
        x["label"] for x in c.get(f"/palette.json?nb={nb}&source=1").json()["commands"]
    ]
    assert current[0] == "Open current PDF in reader: solar.pdf"
    leak = [
        x["label"]
        for x in c.get(f"/palette.json?nb={other}&source=1").json()["commands"]
    ]
    assert not any("solar.pdf" in label for label in leak)  # source of another notebook
    assert c.get("/palette.json?nb=999").status_code == 200  # unknown → global only
    assert c.get("/palette.json").headers["cache-control"] == "no-store"


def test_palette_script_is_safe_and_local():
    js = (STATIC / "app.js").read_text()
    assert (
        ".innerHTML" not in js and "insertAdjacentHTML" not in js and "eval(" not in js
    )
    assert 'fetch("/palette.json?' in js and "textContent" in js
    assert (
        '(e.key === "k" || e.key === "K")' in js
    )  # Ctrl+K; Ctrl+P is left to the browser
    assert '"p"' not in js


def test_body_context_attributes_are_escaped(web):
    html = web["c"].get(f"/notebooks/{web['other']}").text
    assert EVIL not in html and "&lt;script&gt;" in html
    assert f'<body data-notebook-id="{web["other"]}" data-source-id="">' in html


# --- dialogs ------------------------------------------------------------------------


def triggers(html):
    return re.findall(
        r'<a [^>]*data-confirm-action="([^"]+)"[^>]*data-confirm-value="([^"]+)"', html
    )


def test_destructive_actions_use_the_shared_dialog(web):
    c, nb, conv, note, ann = web["c"], web["nb"], web["conv"], web["note"], web["ann"]
    assert (f"/notebooks/{nb}/delete", str(nb)) in triggers(c.get("/").text)
    assert (f"/notebooks/{nb}/sources/1/delete", "1") in triggers(
        c.get(f"/notebooks/{nb}").text
    )
    assert (f"/notebooks/{nb}/sources/1/delete", "1") in triggers(
        c.get(f"/notebooks/{nb}/sources/1").text
    )
    assert (f"/notebooks/{nb}/chat/{conv}/delete", str(conv)) in triggers(
        c.get(f"/notebooks/{nb}/chat/{conv}").text
    )
    assert (f"/notebooks/{nb}/notes/{note}/delete", str(note)) in triggers(
        c.get(f"/notebooks/{nb}/notes/{note}").text
    )
    panel = c.get(f"/notebooks/{nb}/sources/1/annotations", headers=HX).text
    assert (
        f"/notebooks/{nb}/sources/1/annotations/{ann}/delete",
        str(ann),
    ) in triggers(panel)
    row = c.get(f"/notebooks/{nb}").text
    assert 'data-confirm-title="Delete this source permanently?"' in row
    assert 'data-confirm-item="solar.pdf"' in row and 'aria-haspopup="dialog"' in row
    assert 'data-confirm-target="#src-1"' in row
    # without JavaScript the link still opens the server-side confirmation
    assert (
        "Delete this source permanently?"
        in c.get(f"/notebooks/{nb}/sources/1/delete", follow_redirects=True).text
    )


def test_dialog_markup_semantics(web):
    html = web["c"].get("/").text
    m = re.search(
        r'<dialog class="dialog" id="confirm-dialog"(.*?)</dialog>', html, re.S
    )
    assert m
    d = m.group(0)
    assert (
        'aria-labelledby="confirm-dialog-title"' in d
        and 'aria-describedby="confirm-dialog-body"' in d
    )
    assert '<button class="btn" type="button" data-dialog-close>Cancel</button>' in d
    assert d.index("Cancel") < d.index(
        'id="confirm-dialog-submit"'
    )  # focus starts on Cancel
    assert 'name="confirm"' in d
    js = (STATIC / "app.js").read_text()
    assert "showModal()" in js and "lnReturnFocus" in js  # native modal + focus return


def test_dialog_targets_still_require_confirmation_and_isolation(web):
    c, nb, other, conv, note, ann = (
        web[k] for k in ("c", "nb", "other", "conv", "note", "ann")
    )
    for url in (
        f"/notebooks/{nb}/delete",
        f"/notebooks/{nb}/sources/1/delete",
        f"/notebooks/{nb}/chat/{conv}/delete",
        f"/notebooks/{nb}/notes/{note}/delete",
        f"/notebooks/{nb}/sources/1/annotations/{ann}/delete",
    ):
        assert c.post(url, data={"confirm": "wrong"}, headers=HX).status_code == 422, (
            url
        )
    for url in (
        f"/notebooks/{other}/sources/1/delete",
        f"/notebooks/{other}/chat/{conv}/delete",
        f"/notebooks/{other}/notes/{note}/delete",
        f"/notebooks/{other}/sources/1/annotations/{ann}/delete",
    ):
        assert (
            c.post(url, data={"confirm": url.split("/")[-2]}, headers=HX).status_code
            == 404
        ), url


# --- feedback -----------------------------------------------------------------------


def notified(r):
    return json.loads(r.headers.get("x-ln-notify", "null"))


def test_htmx_mutations_send_toasts(web):
    c, nb, ann = web["c"], web["nb"], web["ann"]
    r = c.post("/notebooks", data={"name": "Fresh"}, headers=HX)
    assert notified(r) == {"kind": "success", "message": "Notebook created"}
    r = c.post(
        f"/notebooks/{nb}/sources/1/annotations/{ann}",
        data={"comment": "ok"},
        headers=HX,
    )
    assert notified(r)["message"] == "Annotation updated"
    assert r.headers["HX-Trigger"] == "annotations-changed"  # Phase 8 contract kept
    r = c.post(
        f"/notebooks/{nb}/chat/{web['conv']}/rename",
        data={"title": "Renamed"},
        headers=HX,
    )
    assert notified(r)["message"] == "Conversation renamed"
    r = c.post(f"/notebooks/{nb}/sources/1/delete", data={"confirm": "1"}, headers=HX)
    assert notified(r)["message"] == "Source deleted"
    js = (STATIC / "app.js").read_text()
    assert 'getResponseHeader("X-LN-Notify")' in js and "alert(" not in js


def test_flash_after_redirect_is_shown_once(web):
    c, nb, note = web["c"], web["nb"], web["note"]
    r = c.post(
        f"/notebooks/{nb}/notes/{note}/delete",
        data={"confirm": str(note)},
        headers=ORIGIN,
        follow_redirects=False,
    )
    assert r.status_code == 303 and FLASH_COOKIE in r.headers["set-cookie"]
    assert (
        "HttpOnly" in r.headers["set-cookie"]
        and "SameSite=strict" in r.headers["set-cookie"]
    )
    first = c.get(r.headers["location"])
    assert (
        'class="toast toast-success" data-toast' in first.text
        and "Note deleted" in first.text
    )
    assert 'id="toasts" aria-live="polite"' in first.text
    assert "Note deleted" not in c.get(r.headers["location"]).text  # only once
    c.cookies.set(FLASH_COOKIE, EVIL)  # unknown keys are ignored, never echoed
    assert EVIL not in c.get("/").text and "toast-" not in c.get("/").text
    assert set(FLASH_MESSAGES) >= {
        "note-deleted",
        "source-deleted",
        "conversation-deleted",
    }


# --- accessibility ------------------------------------------------------------------


def test_primary_pages_are_accessible(web):
    for name, url in pages(web).items():
        r = web["c"].get(url)
        assert r.status_code == 200, name
        assert_accessible(r.text, name)


def test_live_regions_and_status(web):
    c, nb = web["c"], web["nb"]
    chat = c.get(f"/notebooks/{nb}/chat/{web['conv']}").text
    log = re.search(r'<div class="chat-log"[^>]*>', chat).group(0)
    assert (
        "aria-live" not in log and 'role="log"' not in log
    )  # no token-by-token chatter
    assert 'id="composer-status" role="status"' in chat
    assert 'id="evidence" aria-label="Evidence" aria-live="polite"' in chat
    notes = c.get(f"/notebooks/{nb}/notes/{web['note']}").text
    assert (
        'id="save-state" role="status"' in notes
        and 'id="note-conflict" role="alert" hidden' in notes
    )
    reader = c.get(f"/notebooks/{nb}/sources/1/read").text
    assert 'id="r-status" role="status"' in reader
    assert 'role="tablist"' in reader and reader.count('role="tab"') == 3
    assert 'role="tabpanel"' in reader and 'role="separator"' in reader
    for label in (
        "Previous page",
        "Next page",
        "Zoom out",
        "Zoom in",
        "Highlight yellow",
        "Highlight pink",
        "Resize side panel",
    ):
        assert f'aria-label="{label}"' in reader
    search = c.get(f"/notebooks/{nb}/search?q=heliostat").text
    assert '<p class="muted search-summary" role="status">' in search


def test_status_is_not_conveyed_by_colour_alone(web):
    c, nb = web["c"], web["nb"]
    badge = c.get("/health/badge").text
    assert "LLM" in badge and ("Connected" in badge or "Unavailable" in badge)
    assert "Embeddings" in badge and "Sleeping" in badge
    panel = c.get(f"/notebooks/{nb}/sources/1/annotations", headers=HX).text
    assert '<span class="ann-color">Yellow</span>' in panel
    css = (STATIC / "app.css").read_text()
    assert ":focus-visible { outline: 2px solid var(--focus)" in css
    assert ".error::before" in css and "prefers-reduced-motion" in css


def test_breadcrumbs_have_no_database_ids(web):
    c, nb, conv, note = web["c"], web["nb"], web["conv"], web["note"]
    reader = c.get(f"/notebooks/{nb}/sources/1/read").text
    crumbs = re.search(r'<nav class="crumbs".*?</nav>', reader, re.S).group(0)
    texts = re.findall(r">([^<>]+)<", crumbs)
    assert [t.strip() for t in texts if t.strip()] == [
        "Notebooks",
        "Research",
        "Sources",
        "solar.pdf",
        "Reader",
    ]
    chat = re.search(
        r'<nav class="crumbs".*?</nav>',
        c.get(f"/notebooks/{nb}/chat/{conv}").text,
        re.S,
    ).group(0)
    assert "How precise is it?" in chat and 'aria-current="page"' in chat
    notes = re.search(
        r'<nav class="crumbs".*?</nav>',
        c.get(f"/notebooks/{nb}/notes/{note}").text,
        re.S,
    ).group(0)
    assert "Reading list" in notes


def test_notebook_nav_offers_last_read_pdf(web):
    c, nb = web["c"], web["nb"]
    assert "tab-reader" not in c.get(f"/notebooks/{nb}").text
    c.post(f"/notebooks/{nb}/sources/1/reading-state", data={"page": 2}, headers=ORIGIN)
    for url in (
        f"/notebooks/{nb}",
        f"/notebooks/{nb}/chat",
        f"/notebooks/{nb}/notes",
        f"/notebooks/{nb}/memory",
        f"/notebooks/{nb}/search",
    ):
        assert (
            f'class="tab-reader" href="/notebooks/{nb}/sources/1/read"'
            in c.get(url).text
        ), url


# --- chat ---------------------------------------------------------------------------


def test_chat_polish_contracts(web):
    c, nb, conv = web["c"], web["nb"], web["conv"]
    html = c.get(f"/notebooks/{nb}/chat/{conv}").text
    assert html.count('data-copy-target=".msg-body"') == 1  # complete answers only
    assert EVIL not in html and "&lt;script&gt;" in html  # copied text is escaped text
    assert 'class="cite"' in html and f"/notebooks/{nb}/chat/citations/" in html
    assert (
        f'data-retry-url="/notebooks/{nb}/chat/messages/{web["failed"]}/retry"' in html
    )
    assert html.count('<time class="msg-time"') == 4
    assert (
        'id="to-latest" hidden' in html and 'aria-describedby="composer-hint"' in html
    )
    assert (
        'data-collapse-key="chat-sidebar"' in html
        and 'aria-controls="conv-panel"' in html
    )
    assert (
        c.post(
            f"/notebooks/{nb}/chat/send", data={"question": "   "}, headers=ORIGIN
        ).status_code
        == 422
    )
    assert (
        c.post(
            f"/notebooks/{nb}/chat/messages/{web['answer']}/stop", headers=ORIGIN
        ).status_code
        == 409
    )
    js = (STATIC / "chat.js").read_text()
    assert "if (active) return;" in js  # no duplicate sends
    assert "nearBottom()" in js and "never yank" in js


# --- notes --------------------------------------------------------------------------


def test_notes_contracts(web):
    c, nb, note = web["c"], web["nb"], web["note"]
    page = c.get(f"/notebooks/{nb}/notes/{note}").text
    assert "Linked evidence" in page and "solar.pdf" in page
    assert 'data-copy-target="note-content"' in page and "data-discard-reload" in page
    ok = c.post(
        f"/notebooks/{nb}/notes/{note}/autosave",
        data={"title": "Reading list", "content": "v2", "revision": 1},
        headers=ORIGIN,
    )
    assert ok.json()["ok"] and ok.json()["revision"] == 2
    stale = c.post(
        f"/notebooks/{nb}/notes/{note}/autosave",
        data={"title": "Reading list", "content": "old", "revision": 1},
        headers=ORIGIN,
    )
    assert stale.status_code == 409 and stale.json()["conflict"] is True
    home = c.get(f"/notebooks/{nb}/notes?new=1").text
    assert re.search(r'id="new-note-title"[^>]*data-n-focus autofocus', home)
    r = c.post(
        f"/notebooks/{nb}/notes",
        data={"title": "Fresh"},
        headers=ORIGIN,
        follow_redirects=False,
    )
    created = c.get(r.headers["location"]).text
    assert "Note created" in created and re.search(
        r'id="note-content"[^>]*autofocus', created
    )


# --- search -------------------------------------------------------------------------


def test_search_contracts_and_no_worker(web):
    c, nb = web["c"], web["nb"]
    for mode in ("all", "sources", "notes", "annotations", "chats", "memory"):
        r = c.get(f"/notebooks/{nb}/search?q=heliostat&mode={mode}&engine=lexical")
        assert r.status_code == 200, mode
    html = c.get(f"/notebooks/{nb}/search?q=heliostat").text
    assert (
        "data-result-nav" in html
        and "data-result-input" in html
        and "data-result>" in html
    )
    assert "Fast keyword" in html and "exact words" in html and "Hybrid" in html
    assert "<mark>" in html
    glob = c.get("/search?q=heliostat").text
    assert (
        "Search all notebooks" in glob and f"/notebooks/{nb}/search?q=heliostat" in glob
    )
    assert "never loads the embedding model" in glob
    nothing = c.get("/search?q=zzzqqq").text
    assert "No matches" in nothing
    xss = c.get("/search", params={"q": EVIL}).text
    assert EVIL not in xss
    assert web["calls"] == []  # Fast keyword search never created the embedder


# --- reader -------------------------------------------------------------------------


def test_reader_contracts(web):
    c, nb, conv = web["c"], web["nb"], web["conv"]
    cit = re.search(
        r"/chat/citations/(\d+)", c.get(f"/notebooks/{nb}/chat/{conv}").text
    ).group(1)
    html = c.get(f"/notebooks/{nb}/sources/1/read?citation={cit}").text
    assert 'data-page="2"' in html and 'data-focus-kind="citation"' in html
    assert (
        f'href="/notebooks/{nb}/chat/{conv}#msg-{web["answer"]}">← Back to chat' in html
    )
    assert 'id="r-back" hidden' in html and 'id="r-fit" aria-pressed="true"' in html
    assert (
        f'class="tab-reader" href="/notebooks/{nb}/sources/1/read" aria-current="page"'
        in html
    )
    plain = c.get(f"/notebooks/{nb}/sources/1/read").text
    assert "Back to chat" not in plain
    js = (STATIC / "reader.js").read_text()
    # split width and side tab live in the browser only; the server keeps page/zoom
    assert 'LN.store.set("reader:side"' in js and 'LN.store.set("reader:tab"' in js
    assert (
        "reading-state" in js
        and "side"
        not in c.post(
            f"/notebooks/{nb}/sources/1/reading-state", data={"page": 1}, headers=ORIGIN
        ).text
    )
    assert ".innerHTML" not in js and 'scrollIntoView({ block: "center" })' not in js


# --- error handling -----------------------------------------------------------------


def test_error_pages(web):
    c, nb = web["c"], web["nb"]
    r = c.get("/no/such/page", headers=PAGE)
    assert (
        r.status_code == 404 and "Not found" in r.text and "Go to notebooks" in r.text
    )
    assert c.get("/no/such/page").json() == {"detail": "Not Found"}  # API callers: JSON
    r = c.get(f"/notebooks/{nb}/sources/42", headers=PAGE)
    assert (
        r.status_code == 404
        and f'href="/notebooks/{nb}">Back to the notebook' in r.text
    )
    r = c.get("/notebooks/31337", headers=PAGE)
    assert r.status_code == 404 and "Back to the notebook" not in r.text
    r = c.get("/notebooks/not-a-number", headers=PAGE)
    assert r.status_code == 422 and "Invalid request" in r.text
    assert c.get("/notebooks/not-a-number").status_code == 422


def test_forbidden_is_a_readable_page(web):
    c = web["c"]
    r = c.post(
        "/notebooks",
        data={"name": "x"},
        headers={"Origin": "http://evil.example", **PAGE},
    )
    assert r.status_code == 403 and "<h1>Not allowed</h1>" in r.text
    assert "evil.example" in r.text and "<script" not in r.text
    r = c.post(
        "/notebooks", data={"name": "x"}, headers={"Origin": "http://evil.example/<b>"}
    )
    assert r.status_code == 403 and r.headers["content-type"].startswith("text/plain")


def test_unexpected_errors_are_sanitised(tmp_path, caplog):
    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "d")})
    app = create_app(
        s,
        env={},
        llm_transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": []})
        ),
    )

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret /home/user/data.sqlite3 SELECT * FROM notes")

    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as c:
        page = c.get("/boom", headers=PAGE)
        assert page.status_code == 500 and "Something went wrong" in page.text
        api = c.get("/boom")
        assert api.status_code == 500 and api.json() == {"error": "Unexpected error"}
        for body in (page.text, api.text):
            assert (
                "secret" not in body
                and "SELECT" not in body
                and "Traceback" not in body
            )
    assert "secret" in caplog.text  # details stay in the local log


# --- lazy loading -------------------------------------------------------------------


def test_navigation_and_palette_never_start_the_worker(tmp_path):
    code = f"""
import sys, httpx
sys.path.insert(0, {str(Path(__file__).parent)!r})
from pathlib import Path
from fastapi.testclient import TestClient
from conftest import FakeEmbedder, make_pdf
from litenotebook.config import load_settings
from litenotebook.db import open_db
from litenotebook.notebooks import create_notebook
from litenotebook.notes import create_note
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.web.app import create_app
s = load_settings({{"LITENOTEBOOK_DATA_DIR": {str(tmp_path / "d")!r}}})
c = open_db(s.db_path)
nb = create_notebook(c, "n")
ingest_pdf(c, s, nb, make_pdf(Path({str(tmp_path)!r}) / "x.pdf", ["Heliostat text."]),
           FakeEmbedder(), FaissStore(c, s))
create_note(c, nb.id, "A note", "heliostat")
c.close()
del sys.modules["faiss"]  # ingestion used FAISS; prove navigation does not
calls = []
def factory():
    calls.append(1)
    raise AssertionError("must not create the embedder")
def llama(r):
    return httpx.Response(200, json={{"data": []}})
app = create_app(s, env={{}}, embedder_factory=factory,
                 llm_transport=httpx.MockTransport(llama))
urls = ["/", "/search", "/search?q=heliostat", "/settings", "/palette.json",
        "/palette.json?nb=1&source=1",
        "/notebooks/1", "/notebooks/1/chat", "/notebooks/1/chat/new",
        "/notebooks/1/notes",
        "/notebooks/1/notes/1", "/notebooks/1/memory", "/notebooks/1/memory/graph",
        "/notebooks/1/memory/graph.json?node=1", "/notebooks/1/search",
        "/notebooks/1/search?q=heliostat&engine=lexical", "/notebooks/1/sources/1",
        "/notebooks/1/sources/1/read", "/health/badge", "/healthz"]
with TestClient(app, base_url="{BASE}") as cl:
    for u in urls:
        assert cl.get(u).status_code == 200, u
heavy = ("torch", "sentence_transformers", "transformers", "faiss")
print("HEAVY", [m for m in heavy if m in sys.modules], len(calls))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "HEAVY [] 0" in out.stdout


def test_no_external_or_unsafe_assets():
    for js in STATIC.glob("*.js"):
        if js.name == "htmx.min.js":
            continue
        text = js.read_text()
        assert ".innerHTML" not in text and "insertAdjacentHTML" not in text, js.name
        urls = re.findall(r"https?://[^\s\"']+", text)
        assert all(u.startswith("http://www.w3.org/") for u in urls), (js.name, urls)
        assert "alert(" not in text, js.name
