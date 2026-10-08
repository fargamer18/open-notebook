"""Phase 2 web app: routes, HTMX fragments, settings, health and local security."""

import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.config import load_settings
from litenotebook.db import connect
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.notebooks import get_notebook_by_name
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
SAME_ORIGIN = {"Origin": BASE}
HX = {**SAME_ORIGIN, "HX-Request": "true"}
SECRET = "sk-test-SECRET-4242"
QWEN = "Qwen/Qwen3-4B-GGUF:Q4_K_M"
WEB_DIR = Path(__file__).resolve().parents[1] / "src" / "litenotebook" / "web"


class FakeLlama:
    """Mock llama-server: only GET /models is allowed (no generation)."""

    def __init__(self, models=(QWEN,), mode="ok"):
        self.models, self.mode, self.requests = list(models), mode, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.method == "GET" and request.url.path.endswith("/models")
        if self.mode == "down":
            raise httpx.ConnectError("Connection refused", request=request)
        if self.mode == "500":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})


@pytest.fixture
def llama():
    return FakeLlama()


@pytest.fixture
def app_settings(tmp_path):
    return load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})


def make_client(app_settings, llama, env=None):
    app = create_app(
        app_settings, env=env or {}, llm_transport=httpx.MockTransport(llama)
    )
    return TestClient(app, base_url=BASE)


@pytest.fixture
def client(app_settings, llama):
    return make_client(app_settings, llama)


def create(client, name, description="", hx=False):
    return client.post(
        "/notebooks",
        data={"name": name, "description": description},
        headers=HX if hx else SAME_ORIGIN,
        follow_redirects=False,
    )


def notebook_id(app_settings, name):
    conn = connect(app_settings.db_path)
    try:
        return get_notebook_by_name(conn, name).id
    finally:
        conn.close()


def populate(app_settings, name, tmp_path):
    """Give a notebook a real ingested source (files + FAISS index on disk)."""
    conn = connect(app_settings.db_path)
    try:
        nb = get_notebook_by_name(conn, name)
        pdf = make_pdf(
            tmp_path / f"{name}.pdf", ["Rank equals pivots.", "Second page."]
        )
        ingest_pdf(
            conn, app_settings, nb, pdf, FakeEmbedder(), FaissStore(conn, app_settings)
        )
        return nb.id
    finally:
        conn.close()


# --- startup & shell ---------------------------------------------------------------


def test_startup_creates_data_dir_and_schema(app_settings, llama):
    assert not app_settings.data_dir.exists()
    make_client(app_settings, llama)
    assert app_settings.db_path.exists()


def test_lifespan_keeps_wal_file_between_requests(app_settings, llama):
    wal = app_settings.db_path.with_name(app_settings.db_path.name + "-wal")
    with make_client(app_settings, llama) as c:  # runs the lifespan
        assert "One" in c.post("/notebooks", data={"name": "One"}).text
        assert wal.exists()  # not deleted when the request's connection closed
    assert c.get("/").status_code == 200


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>Notebooks · LiteNotebook</title>" in r.text
    assert '<script src="/static/htmx.min.js" defer></script>' in r.text
    assert 'href="/settings"' in r.text and "No notebooks yet" in r.text


def test_static_assets_are_local(client):
    assert client.get("/static/htmx.min.js").status_code == 200
    assert "background" in client.get("/static/app.css").text


def test_no_external_urls_in_templates_or_css():
    pattern = re.compile(
        r"""(src|href|action)\s*=\s*["']\s*(https?:)?//|url\(\s*["']?(https?:)?//|@import"""
    )
    offenders = [
        p.name
        for p in WEB_DIR.rglob("*")
        if p.suffix in (".html", ".css") and pattern.search(p.read_text())
    ]
    assert offenders == []


# --- notebooks -----------------------------------------------------------------------


def test_create_notebook_plain_form_redirects(client):
    r = create(client, "Linear algebra", "GATE notes")
    assert r.status_code == 303 and r.headers["location"] == "/"
    page = client.get("/").text
    assert (
        "Linear algebra" in page
        and "GATE notes" in page
        and "0 sources · 0 chunks" in page
    )


def test_create_notebook_htmx_returns_row_out_of_band(client):
    r = create(client, "Physics", hx=True)
    assert r.status_code == 200
    assert 'hx-swap-oob="afterbegin:#notebook-list"' in r.text
    assert 'id="notebook-empty" hidden hx-swap-oob="true"' in r.text
    assert "Created “Physics”" in r.text and 'value=""' in r.text  # form cleared


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "cannot be empty"),
        ("   ", "cannot be empty"),
        ("x" * 101, "at most 100"),
        ("bad\x07name", "control characters"),
    ],
)
def test_create_invalid_name(client, name, message):
    for hx in (False, True):
        r = create(client, name, hx=hx)
        assert r.status_code == 422 and message in r.text
    assert "No notebooks yet" in client.get("/").text


def test_create_duplicate_name(client):
    create(client, "Research")
    r = create(client, "research", hx=True)
    assert r.status_code == 422 and "already exists" in r.text
    assert 'value="research"' in r.text  # input preserved


def test_rename_notebook(client, app_settings):
    create(client, "Draft")
    nb = notebook_id(app_settings, "Draft")
    edit = client.get(f"/notebooks/{nb}/rename", headers=HX)
    assert (
        edit.status_code == 200
        and 'name="name"' in edit.text
        and 'value="Draft"' in edit.text
    )
    r = client.post(f"/notebooks/{nb}/rename", data={"name": "Final"}, headers=HX)
    assert r.status_code == 200 and "Final" in r.text and 'class="nb-row"' in r.text
    assert "Final" in client.get(f"/notebooks/{nb}").text


def test_rename_errors(client, app_settings):
    create(client, "One")
    create(client, "Two")
    nb = notebook_id(app_settings, "One")
    r = client.post(f"/notebooks/{nb}/rename", data={"name": "TWO"}, headers=HX)
    assert (
        r.status_code == 422 and "already exists" in r.text and "is-editing" in r.text
    )
    r = client.post(f"/notebooks/{nb}/rename", data={"name": ""}, headers=SAME_ORIGIN)
    assert r.status_code == 422 and "cannot be empty" in r.text
    assert (
        client.post(
            "/notebooks/999/rename", data={"name": "x"}, headers=SAME_ORIGIN
        ).status_code
        == 404
    )


def test_delete_requires_confirmation_and_shows_preview(client, app_settings, tmp_path):
    create(client, "Research")
    nb = populate(app_settings, "Research", tmp_path)
    preview = client.get(f"/notebooks/{nb}/delete", headers=HX)
    assert preview.status_code == 200
    text = re.sub(r"\s+", " ", preview.text)
    assert "<dt>Notebook</dt><dd>Research</dd>" in text.replace("> <", "><")
    assert "<dt>Sources</dt><dd>1</dd>" in text.replace("> <", "><")
    assert "<dt>Chunks</dt><dd>2</dd>" in text.replace("> <", "><")
    assert f'name="confirm" value="{nb}"' in preview.text

    for confirm in ("", "yes", str(nb + 1)):
        r = client.post(
            f"/notebooks/{nb}/delete", data={"confirm": confirm}, headers=HX
        )
        assert r.status_code == 422 and "not confirmed" in r.text
    assert app_settings.notebook_dir(nb).exists()
    assert "Research" in client.get("/").text

    # non-HTMX confirmation page works too
    assert (
        "Delete this notebook permanently?"
        in client.get(f"/notebooks/{nb}/delete").text
    )


def test_delete_notebook_removes_rows_and_files(client, app_settings, tmp_path):
    for name in ("Keep", "Remove"):
        create(client, name)
        populate(app_settings, name, tmp_path)
    keep, remove = (
        notebook_id(app_settings, "Keep"),
        notebook_id(app_settings, "Remove"),
    )

    r = client.post(
        f"/notebooks/{remove}/delete", data={"confirm": str(remove)}, headers=HX
    )
    assert (
        r.status_code == 200
        and "Deleted “Remove” — 1 sources and 2 chunks removed" in r.text
    )

    assert not app_settings.notebook_dir(remove).exists()
    assert app_settings.faiss_index_path(keep).exists()
    conn = connect(app_settings.db_path)
    try:
        assert (
            conn.execute(
                "SELECT count(*) FROM sources WHERE notebook_id = ?", (remove,)
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM chunks WHERE notebook_id = ?", (keep,)
            ).fetchone()[0]
            == 2
        )
        assert FaissStore(conn, app_settings).check(keep).ok
    finally:
        conn.close()
    assert client.get(f"/notebooks/{remove}").status_code == 404
    assert (
        client.post(
            f"/notebooks/{remove}/delete",
            data={"confirm": str(remove)},
            headers=SAME_ORIGIN,
        ).status_code
        == 404
    )


def test_notebook_isolation_in_list_and_workspace(client, app_settings, tmp_path):
    create(client, "Alpha")
    create(client, "Beta")
    populate(app_settings, "Alpha", tmp_path)
    alpha, beta = notebook_id(app_settings, "Alpha"), notebook_id(app_settings, "Beta")
    assert "<dd>2</dd>" in client.get(f"/notebooks/{alpha}").text.replace(" ", "")
    beta_page = client.get(f"/notebooks/{beta}").text
    assert "Beta" in beta_page and "Alpha</h1>" not in beta_page
    assert re.search(r"Sources</dt>\s*<dd>0</dd>", beta_page)


def test_workspace(client, app_settings):
    create(client, "Shell", "workspace description")
    nb = notebook_id(app_settings, "Shell")
    r = client.get(f"/notebooks/{nb}")
    assert r.status_code == 200
    for text in (
        "Shell",
        "workspace description",
        "all-MiniLM-L6-v2 · 384-d",
        "Sources",
        "Chat",
        "Notes",
        "Search",
        "Arrives in Phase 3".lower(),
    ):
        assert text.lower() in r.text.lower()
    chat = client.get(f"/notebooks/{nb}?tab=chat").text
    assert re.search(r'tab=chat" aria-current="page"', chat)
    assert "Phase 5" in chat
    assert client.get(f"/notebooks/{nb}?tab=../../etc").status_code == 200  # falls back
    assert client.get("/notebooks/424242").status_code == 404
    assert client.get("/notebooks/not-a-number").status_code == 422


# --- settings -----------------------------------------------------------------


def save(client, hx=True, **fields):
    data = {
        "llm_base_url": "http://127.0.0.1:8080/v1",
        "llm_model": QWEN,
        "rag_top_k": "8",
        "llm_api_key": "",
        **fields,
    }
    return client.post(
        "/settings",
        data=data,
        headers=HX if hx else SAME_ORIGIN,
        follow_redirects=False,
    )


def test_settings_page_defaults(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert 'value="http://127.0.0.1:8080/v1"' in r.text
    assert f'value="{QWEN}"' in r.text and 'value="8"' in r.text
    assert "Not set (not needed for a local llama-server)" in r.text


def test_settings_persist(client, app_settings, llama):
    assert save(client, hx=False, rag_top_k="5", llm_model="other").status_code == 303
    page = make_client(app_settings, llama).get("/settings").text  # fresh app, same DB
    assert 'value="5"' in page and 'value="other"' in page


def test_settings_validation_errors(client):
    r = save(client, llm_base_url="ftp://x", rag_top_k="0")
    assert r.status_code == 422
    assert "http:// or https://" in r.text and "between 1 and 20" in r.text
    assert 'value="ftp://x"' in r.text  # user input kept for correction
    assert 'value="8"' in client.get("/settings").text  # nothing saved


def test_api_key_is_never_returned(client, llama):
    r = save(client, llm_api_key=SECRET)
    assert r.status_code == 200 and SECRET not in r.text
    assert "Configured — leave blank to keep it" in r.text
    for response in (
        client.get("/settings"),
        client.get("/healthz"),
        client.get("/health/badge"),
        client.post("/settings/test", headers=HX),
        client.post("/settings/models", headers=HX),
        client.post("/settings/test", headers=SAME_ORIGIN),
    ):
        assert SECRET not in response.text
    # ...but it is sent to the LLM server
    assert llama.requests[-1].headers["authorization"] == f"Bearer {SECRET}"
    # blank field keeps it; checkbox clears it
    save(client)
    assert llama.requests and "Configured" in client.get("/settings").text
    save(client, clear_api_key="1")
    assert "Not set" in client.get("/settings").text


def test_environment_overrides(app_settings, llama):
    env = {
        "LITENOTEBOOK_LLM_BASE_URL": "http://localhost:9999/v1",
        "LITENOTEBOOK_LLM_API_KEY": SECRET,
    }
    client = make_client(app_settings, llama, env=env)
    page = client.get("/settings").text
    assert 'value="http://localhost:9999/v1"' in page
    assert 'Set by <span class="mono">LITENOTEBOOK_LLM_BASE_URL</span>' in page
    assert re.search(r'id="llm_base_url"[^>]*disabled', page, re.S)
    assert SECRET not in page

    save(client, llm_base_url="http://127.0.0.1:1111/v1", llm_api_key="other-key")
    conn = connect(app_settings.db_path)
    try:
        stored = dict(conn.execute("SELECT key, value FROM settings").fetchall())
    finally:
        conn.close()
    assert "llm.base_url" not in stored and "llm.api_key" not in stored
    assert SECRET not in stored.values()
    client.post("/settings/test", headers=HX)
    assert str(llama.requests[-1].url) == "http://localhost:9999/v1/models"
    assert llama.requests[-1].headers["authorization"] == f"Bearer {SECRET}"


def test_model_discovery_lists_models_without_changing_selection(app_settings):
    llama = FakeLlama(models=["model-a", QWEN, "model-b"])
    client = make_client(app_settings, llama)
    r = client.post("/settings/models", headers=HX)
    assert r.status_code == 200
    for m in ("model-a", "model-b"):
        assert m in r.text and f'<option value="{m}">' in r.text
    assert re.search(re.escape(QWEN) + r'\s*<span class="tag">selected</span>', r.text)
    assert 'id="model-options" hx-swap-oob="true"' in r.text
    assert f'value="{QWEN}"' in client.get("/settings").text


def test_connection_test_success(client):
    r = client.post("/settings/test", headers=HX)
    assert "status-ok" in r.text and f"Connected — {QWEN} available" in r.text


def test_connection_test_failure(app_settings):
    r = make_client(app_settings, FakeLlama(mode="down")).post(
        "/settings/test", headers=HX
    )
    assert r.status_code == 200
    assert (
        "status-err" in r.text and "Unreachable" in r.text and "llama-server" in r.text
    )


def test_connection_test_http_error(app_settings):
    r = make_client(app_settings, FakeLlama(mode="500")).post(
        "/settings/test", headers=HX
    )
    assert "status-warn" in r.text and "HTTP 500" in r.text


def test_selected_model_not_found(app_settings):
    client = make_client(app_settings, FakeLlama(models=["some-other-model"]))
    r = client.post("/settings/test", headers=HX)
    assert "status-warn" in r.text and "was not found" in r.text
    assert "some-other-model" in r.text


def test_connection_test_without_javascript_renders_full_page(client):
    r = client.post("/settings/test", headers=SAME_ORIGIN)
    assert r.status_code == 200 and "<html" in r.text and "Connected" in r.text


# --- health -------------------------------------------------------------------


def test_health_llm_online(client):
    h = client.get("/healthz").json()
    assert h["status"] == "ok"
    assert h["database"] == {"status": "ok", "schema_version": 2}
    assert h["data"]["status"] == "ok"
    assert h["llm"]["status"] == "reachable" and h["llm"]["model_available"] is True
    assert "LLM online" in client.get("/health/badge").text


def test_health_llm_offline_app_still_works(app_settings):
    client = make_client(app_settings, FakeLlama(mode="down"))
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["llm"]["status"] == "unreachable"
    assert "LLM offline" in client.get("/health/badge").text
    assert create(client, "Works offline").status_code == 303
    assert save(client).status_code == 200
    assert client.get("/settings").status_code == 200


def test_health_reports_storage_problems(client, app_settings):
    app_settings.data_dir.chmod(0o500)
    try:
        r = client.get("/healthz")
        assert r.status_code == 503 and r.json()["data"]["status"] == "error"
    finally:
        app_settings.data_dir.chmod(0o700)


# --- local security -----------------------------------------------------------


@pytest.mark.parametrize(
    "host", ["evil.example", "127.0.0.1.nip.io", "192.168.1.5:5056", ""]
)
def test_rejects_foreign_host_headers(client, host):
    assert client.get("/", headers={"Host": host}).status_code == 400


@pytest.mark.parametrize(
    "host", ["127.0.0.1:5056", "localhost:5056", "[::1]:5056", "localhost"]
)
def test_accepts_loopback_hosts(client, host):
    assert client.get("/", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Referer": "http://evil.example/page"},
        {"Referer": "http://127.0.0.1:5056.evil.example/"},
    ],
)
def test_rejects_cross_origin_state_changes(client, headers):
    for path, data in (
        ("/notebooks", {"name": "pwned"}),
        ("/settings", {"rag_top_k": "1"}),
        ("/settings/test", {}),
    ):
        assert client.post(path, data=data, headers=headers).status_code == 403
    assert "pwned" not in client.get("/").text


def test_allows_same_origin_and_non_browser_clients(client):
    assert create(client, "same-origin").status_code == 303
    r = client.post("/notebooks", data={"name": "curl"}, follow_redirects=False)
    assert r.status_code == 303  # no Origin/Referer/Sec-Fetch-Site: not a browser
    r = client.post(
        "/notebooks",
        data={"name": "ref"},
        follow_redirects=False,
        headers={"Referer": f"{BASE}/", "Sec-Fetch-Site": "same-origin"},
    )
    assert r.status_code == 303


def test_safe_methods_allowed_cross_origin(client):
    assert client.get("/", headers={"Origin": "http://evil.example"}).status_code == 200


def test_security_headers(client):
    headers = client.get("/").headers
    assert "script-src 'self'" in headers["content-security-policy"]
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"
    assert (
        client.get("/", headers={"Host": "evil.example"}).headers["x-frame-options"]
        == "DENY"
    )


def test_no_inline_scripts_or_styles_in_pages(client, app_settings):
    create(client, "x")
    nb = notebook_id(app_settings, "x")
    for path in ("/", "/settings", f"/notebooks/{nb}", f"/notebooks/{nb}/delete"):
        html = client.get(path).text
        assert re.findall(r"<script(?![^>]*\bsrc=)", html) == []
        assert (
            " style=" not in html
            and "<style" not in html
            and " on" + "click" not in html
        )


def test_html_is_escaped_everywhere(client, app_settings):
    evil = '<script>alert("x")</script>'
    create(client, evil, description='<img src=x onerror="alert(1)">')
    nb = notebook_id(app_settings, evil)
    pages = [
        client.get("/").text,
        client.get(f"/notebooks/{nb}").text,
        client.get(f"/notebooks/{nb}/delete", headers=HX).text,
        client.get(f"/notebooks/{nb}/rename", headers=HX).text,
        create(client, evil, hx=True).text,  # duplicate error echoes the name
    ]
    for html in pages:
        assert "<script>alert" not in html and "<img src=x" not in html
        assert "&lt;script&gt;" in html or "&#34;" in html or "&lt;" in html


def test_web_pages_do_not_load_minilm_or_torch(tmp_path):
    """Browsing must never trigger the slow MiniLM/torch import."""
    code = f"""
import sys, httpx
from fastapi.testclient import TestClient
from litenotebook.config import load_settings
from litenotebook.web.app import create_app
settings = load_settings({{"LITENOTEBOOK_DATA_DIR": {str(tmp_path / "d")!r}}})
fake = httpx.MockTransport(lambda r: httpx.Response(200, json={{"data": []}}))
app = create_app(settings, env={{}}, llm_transport=fake)
c = TestClient(app, base_url="{BASE}")
h = {{"Origin": "{BASE}"}}
c.post("/notebooks", data={{"name": "n"}}, headers=h)
pages = ("/", "/notebooks/1", "/notebooks/1/delete", "/settings", "/healthz",
         "/health/badge")
for p in pages:
    assert c.get(p).status_code == 200, p
c.post("/settings/test", headers=h)
c.post("/notebooks/1/delete", data={{"confirm": "1"}}, headers=h)
heavy = [m for m in ("torch", "sentence_transformers", "transformers", "faiss")
         if m in sys.modules]
print("HEAVY", heavy)
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "HEAVY []" in out.stdout
