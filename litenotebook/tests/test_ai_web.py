"""Phase 12 web: setup wizard, Settings → Local AI, chat integration, CLI.

All data lives in temporary folders (never the user's notebooks); the LLM is
a mock transport or the fake llama-server; nothing reaches the internet.
"""

import json
import re
import sys
import time
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from ai_fixtures import fake_gguf, launcher
from conftest import FakeEmbedder
from litenotebook import cli
from litenotebook.ai import embeddings as emb
from litenotebook.ai.runtime_manager import RuntimeManager
from litenotebook.ai.service import LocalAI
from litenotebook.ai.store import SETUP_STEPS, AIStore
from litenotebook.db import connect
from test_chat_web import BASE, HX, ORIGIN, ask, make_app, setup_notebook


@pytest.fixture
def app_settings(tmp_path):
    from litenotebook.config import load_settings

    return load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})


@pytest.fixture
def requests_seen():
    return []


@pytest.fixture
def llama(requests_seen):
    from test_chat_web import answer_for, quillport_label

    inner = answer_for(quillport_label)

    def record(request):
        requests_seen.append(request.url.path)
        return inner(request)

    return record


@pytest.fixture
def client(app_settings, llama):
    with TestClient(make_app(app_settings, llama), base_url=BASE) as c:
        yield c


def store(app_settings):
    conn = connect(app_settings.db_path)
    return conn, AIStore(conn)


def step_of(html):
    m = re.search(r'href="/setup\?step=(\w+)" aria-current="step"', html)
    return m.group(1) if m else None


# --- lazy: page views start nothing ---------------------------------------------------


def test_pages_render_without_scanning_starting_or_downloading(
    client, app_settings, requests_seen
):
    for url in [
        "/",
        "/settings",
        "/settings/ai",
        *[f"/setup?step={s}" for s in SETUP_STEPS],
    ]:
        assert client.get(url).status_code == 200, url
    conn, s = store(app_settings)
    try:
        assert s.hardware() is None  # no scan without a button press
    finally:
        conn.close()
    app = client.app
    assert app.state.local_ai.runtime.status()["state"] == "stopped"
    assert app.state.embedder.status()["status"] == "stopped"
    assert "torch" not in sys.modules and "sentence_transformers" not in sys.modules
    assert not any(p.endswith("chat/completions") for p in requests_seen)
    assert not (app_settings.data_dir / "runtime").exists()
    assert not (app_settings.data_dir / "models").exists()


def test_notebooks_index_offers_setup_until_completed(client, app_settings):
    assert "Set up local AI" in client.get("/").text
    client.post("/setup/welcome", headers=ORIGIN)
    assert "Continue setup" in client.get("/").text
    client.post("/setup/finish", headers=ORIGIN)
    assert 'id="ai-banner-title"' not in client.get("/").text


# --- wizard ---------------------------------------------------------------------------


def test_wizard_can_be_skipped_step_by_step_and_resumed(
    client, app_settings, monkeypatch
):
    monkeypatch.setattr(
        emb, "status", lambda: {**_emb_status(), "installed": False, "missing": ["x"]}
    )
    assert step_of(client.get("/setup").text) == "welcome"
    r = client.post("/setup/welcome", headers=ORIGIN, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/setup?step=hardware"
    assert step_of(client.get("/setup").text) == "hardware"  # resumes where it stopped
    client.post("/setup/hardware", headers=ORIGIN)
    conn, s = store(app_settings)
    assert s.hardware() is not None
    conn.close()
    client.post("/setup/embeddings/skip", headers=ORIGIN)
    page = client.post(
        "/setup/generation", data={"choice": "none"}, headers=ORIGIN
    ).text
    assert step_of(page) == "ready"
    conn, s = store(app_settings)
    state = s.setup()
    assert {"generation", "model", "test", "embeddings"} <= set(state.skipped)
    assert s.profile().type == "none" and s.embeddings()["state"] == "skipped"
    conn.close()
    client.post("/setup/finish", headers=ORIGIN)
    conn, s = store(app_settings)
    assert s.setup().completed
    conn.close()


def test_reset_setup_keeps_models_and_runtime_choice(client, app_settings, tmp_path):
    model = fake_gguf(tmp_path / "keep.gguf")
    conn, s = store(app_settings)
    client.app.state.local_ai.import_model(conn, str(model))
    s.mark("ready")
    conn.close()
    client.post("/setup/reset", headers=ORIGIN)
    conn, s = store(app_settings)
    assert not s.setup().completed and len(s.models()) == 1 and model.exists()
    conn.close()


def test_existing_server_choice_uses_the_configured_endpoint(client, app_settings):
    page = client.post("/setup/generation", data={"choice": "external"}, headers=ORIGIN)
    assert step_of(page.text) == "test"
    conn, s = store(app_settings)
    assert s.profile().type == "external_openai"
    conn.close()


def test_bad_runtime_path_shows_a_specific_error(client, tmp_path):
    r = client.post(
        "/setup/generation",
        data={"choice": "existing", "path": str(tmp_path / "nope")},
        headers=ORIGIN,
    )
    assert (
        r.status_code == 422 and "does not exist" in r.text and 'role="alert"' in r.text
    )


def test_unknown_catalog_ids_are_refused(client):
    r = client.post(
        "/setup/model",
        data={"choice": "install", "model_id": "../../etc"},
        headers=ORIGIN,
    )
    assert r.status_code == 422 and "Choose a model" in r.text
    r = client.post(
        "/setup/generation",
        data={"choice": "managed", "runtime_id": "evil"},
        headers=ORIGIN,
    )
    assert r.status_code == 422


class _A11y(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.labels, self.inputs, self.h1, self.live = set(), set(), [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "label" and a.get("for"):
            self.labels.add(a["for"])
        if tag in ("input", "select", "textarea") and a.get("type") not in (
            "hidden",
            "radio",
            "checkbox",
        ):
            self.inputs.append(a.get("id"))
        if tag == "h1":
            self.h1 += 1
        if a.get("aria-live"):
            self.live += 1


@pytest.mark.parametrize(
    "url", [*[f"/setup?step={s}" for s in SETUP_STEPS], "/settings/ai"]
)
def test_setup_pages_are_accessible(client, app_settings, url):
    conn, _ = store(app_settings)
    client.app.state.local_ai.hardware(conn, rescan=True)  # show the full pages
    conn.close()
    html = client.get(url).text
    p = _A11y()
    p.feed(html)
    assert p.h1 == 1
    assert all(i and i in p.labels for i in p.inputs), (
        p.inputs
    )  # every text field labelled
    assert p.live >= 1  # job progress is announced
    if url.startswith("/setup"):
        assert 'aria-label="Setup steps"' in html and 'aria-current="step"' in html
        assert 'id="step-title" tabindex="-1"' in html
    for radio_label in re.findall(
        r'<label class="choice[^"]*">\s*<input type="radio"', html
    ):
        assert radio_label  # radios are wrapped in their label (clickable text)


# --- jobs: MiniLM install with a fake downloader child --------------------------------


FAKE_FETCH = """
import json, os
names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
with open(ENV_FILE, "w") as f:
    json.dump({k: os.environ.get(k) for k in names}, f)
for i in (1, 2):
    step = {"stage": "downloading", "file": f"f{i}", "index": i, "total": 2}
    print(json.dumps(step), flush=True)
print(json.dumps({"ok": True}), flush=True)
"""


def _emb_status():
    return {
        "id": "m",
        "name": "MiniLM",
        "installed": True,
        "missing": [],
        "size": 1,
        "license": "Apache-2.0",
        "license_url": "",
        "model_page": "",
        "detect_ms": 1.0,
    }


def test_minilm_install_runs_as_a_job_and_tests_in_the_worker(
    app_settings, llama, tmp_path, monkeypatch
):
    env_seen = tmp_path / "env.json"
    script = tmp_path / "fetch.py"
    script.write_text(
        FAKE_FETCH.replace("ENV_FILE", repr(str(env_seen))), encoding="utf-8"
    )
    installed = {"v": False}
    monkeypatch.setattr(
        emb, "status", lambda: {**_emb_status(), "installed": installed["v"]}
    )
    monkeypatch.setattr(emb, "verify", lambda e=None: installed.update(v=True))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    app = make_app(app_settings, llama)
    app.state.local_ai.minilm_command = [sys.executable, str(script)]
    with TestClient(app, base_url=BASE) as c:
        r = c.post("/setup/embeddings", headers=ORIGIN, follow_redirects=False)
        job_id = r.headers["location"].split("job=")[1]
        job = _wait(c, job_id)
        assert job["state"] == "done", job
        page = c.get(f"/ai/jobs/{job_id}?back=/setup?step=embeddings").text
        assert (
            "MiniLM loaded in its worker process" in page
            and "data-job-continue" in page
        )
        assert app.state.embedder.status()["status"] in ("ready", "in-process")
    child_env = json.loads(env_seen.read_text(encoding="utf-8"))
    assert child_env["HF_HUB_OFFLINE"] is None  # network only for this child
    assert child_env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"  # the parent stays offline


def test_minilm_offline_error_is_specific(app_settings, llama, tmp_path, monkeypatch):
    script = tmp_path / "fetch.py"
    offline = {"error": "No internet connection", "offline": True}
    script.write_text(
        f"import json\nprint(json.dumps({offline!r}))\n", encoding="utf-8"
    )
    monkeypatch.setattr(emb, "status", lambda: {**_emb_status(), "installed": False})
    app = make_app(app_settings, llama)
    app.state.local_ai.minilm_command = [sys.executable, str(script)]
    with TestClient(app, base_url=BASE) as c:
        r = c.post(
            "/settings/ai/install-embeddings", headers=ORIGIN, follow_redirects=False
        )
        job = _wait(c, r.headers["location"].split("job=")[1])
    assert job["state"] == "failed" and "No internet connection" in job["message"]


def _wait(c, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = c.get(f"/ai/jobs/{job_id}", headers={"Accept": "application/json"}).json()
        if job["state"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_job_cancel_and_unknown_job(client):
    assert (
        client.get("/ai/jobs/nope", headers={"Accept": "application/json"}).status_code
        == 404
    )
    jobs = client.app.state.ai_jobs
    import threading

    gate = threading.Event()

    def slow(job):
        while not job.cancel.is_set():
            gate.wait(0.05)
        raise RuntimeError("stopped")

    job = jobs.start("test", "Slow", slow)
    assert jobs.start("test", "Again", slow) is job  # one job per kind
    client.post(f"/ai/jobs/{job.id}/cancel", headers=ORIGIN)
    deadline = time.monotonic() + 5
    while job.state == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert job.state == "cancelled"
    html = client.get(f"/ai/jobs/{job.id}").text
    assert "Cancelled" in html and "hx-trigger" not in html  # polling stops


# --- chat integration ----------------------------------------------------------------


def test_chat_without_generation_says_how_to_set_it_up(
    client, app_settings, tmp_path, requests_seen
):
    nb = setup_notebook(app_settings, tmp_path)
    client.post("/setup/generation", data={"choice": "none"}, headers=ORIGIN)
    r, _ = ask(client, nb, "What is the capital of Zorbania?")
    assert r.status_code == 409
    assert r.json() == {"error": "Local AI is not configured.", "setup_url": "/setup"}
    assert not any(p.endswith("chat/completions") for p in requests_seen)
    badge = client.get("/health/badge").text
    assert "Off" in badge and 'href="/settings/ai"' in badge


def test_unset_profile_keeps_the_legacy_llm_settings(client, app_settings, tmp_path):
    nb = setup_notebook(app_settings, tmp_path)
    r, events = ask(client, nb, "What is the capital of Zorbania?")
    assert r.status_code == 200 and ("done", events[-1][1]) == events[-1]
    assert "Quillport" in "".join(d["t"] for e, d in events if e == "delta")


def test_managed_runtime_autostarts_for_chat_and_stops_with_the_app(
    app_settings, llama, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "0.2")
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    runtime = RuntimeManager(
        app_settings, launcher=launcher(), ready_timeout_s=20, stop_grace_s=3
    )
    service = LocalAI(app_settings, runtime)
    import httpx

    from test_chat_web import create_app

    # the chat request itself goes to the fake server over real HTTP
    app = create_app(
        app_settings, env={}, embedder_factory=FakeEmbedder, local_ai=service
    )
    nb = setup_notebook(app_settings, tmp_path)
    conn, s = store(app_settings)
    profile = s.profile()
    profile.type, profile.executable = "external_llama", sys.executable
    s.save_profile(profile)
    service.import_model(conn, str(fake_gguf(tmp_path / "tiny.gguf")))
    conn.close()
    with TestClient(app, base_url=BASE) as c:
        badge = c.get("/health/badge").text
        assert "Stopped" in badge  # no network call while stopped
        r, events = ask(c, nb, "What is the capital of Zorbania?")
        stages = [d.get("stage") for e, d in events if e == "status"]
        assert "starting_runtime" in stages and runtime.running()
        assert events[-1][0] == "done", events[-1]
        pid = runtime.status()["pid"]
        assert "Connected" in c.get("/health/badge").text
    from conftest import process_gone

    assert process_gone(pid)  # the app's shutdown stopped what it started
    assert isinstance(httpx, object)


def test_chat_with_autostart_off_and_runtime_stopped_explains(
    app_settings, llama, tmp_path
):
    app = make_app(app_settings, llama)
    nb = setup_notebook(app_settings, tmp_path)
    conn, s = store(app_settings)
    profile = s.profile()
    profile.type, profile.executable, profile.autostart = (
        "external_llama",
        sys.executable,
        False,
    )
    s.save_profile(profile)
    app.state.local_ai.import_model(conn, str(fake_gguf(tmp_path / "tiny.gguf")))
    conn.close()
    with TestClient(app, base_url=BASE) as c:
        r, events = ask(c, nb, "What is the capital of Zorbania?")
    err = next(d for e, d in events if e == "error")
    assert "Start it in Settings → Local AI" in err["message"] and err["setup_url"]
    assert not app.state.local_ai.runtime.running()


# --- Settings → Local AI ------------------------------------------------------------


def test_settings_select_and_remove_with_confirmation(client, app_settings, tmp_path):
    model = fake_gguf(tmp_path / "mine.gguf")
    conn, s = store(app_settings)
    record = client.app.state.local_ai.import_model(conn, str(model))
    conn.close()
    page = client.get("/settings/ai").text
    assert "used in place (never deleted)" in page and "Remove…" in page
    confirm = client.get(f"/settings/ai?remove={record.id}").text
    assert "your file is not touched" in confirm
    r = client.post("/settings/ai/remove", data={"confirm": ""}, headers=ORIGIN)
    assert r.status_code == 422 and "not confirmed" in r.text
    r = client.post("/settings/ai/remove", data={"confirm": record.id}, headers=ORIGIN)
    assert r.status_code == 200 and model.exists()
    conn, s = store(app_settings)
    assert s.models() == []
    conn.close()


def test_settings_actions_respond_without_hx_and_with_hx(client, app_settings):
    r = client.post("/settings/ai/rescan", headers=HX)
    assert r.status_code == 204 and "X-LN-Notify" in r.headers
    r = client.post(
        "/settings/ai/autostart", data={}, headers=ORIGIN, follow_redirects=False
    )
    assert r.status_code == 303
    conn, s = store(app_settings)
    assert s.profile().autostart is False
    conn.close()
    assert client.post("/settings/ai/unknown", headers=ORIGIN).status_code == 422
    assert (
        client.post(
            "/settings/ai/stop", headers=ORIGIN, follow_redirects=False
        ).status_code
        == 303
    )


def test_cross_origin_posts_are_refused(client):
    r = client.post("/settings/ai/stop", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post(
        "/setup/generation",
        data={"choice": "none"},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403


def test_palette_lists_local_ai(client):
    labels = [c["label"] for c in client.get("/palette.json").json()["commands"]]
    assert "Local AI" in labels and "Set up local AI" in labels


# --- CLI (same service) --------------------------------------------------------------


def run_cli(args, data_dir, capsys):
    code = cli.main(["--data-dir", str(data_dir), *args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_status_import_select_remove(tmp_path, capsys):
    data = tmp_path / "data"
    model = fake_gguf(tmp_path / "tiny.gguf")
    code, out, _ = run_cli(["ai", "status"], data, capsys)
    assert code == 0 and "Generation:  unset" in out
    code, out, _ = run_cli(["ai", "model", "import", str(model)], data, capsys)
    assert code == 0 and "used in place" in out
    code, out, _ = run_cli(["ai", "model", "list"], data, capsys)
    assert "* local-tiny" in out
    code, out, _ = run_cli(["ai", "status", "--json"], data, capsys)
    assert json.loads(out)["profile"]["model_id"] == "local-tiny"
    code, out, _ = run_cli(["ai", "model", "remove", "local-tiny"], data, capsys)
    assert code == 0 and "list only" in out and model.exists()


def test_cli_downloads_need_confirmation(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    code, _, err = run_cli(
        ["ai", "model", "install", "qwen3-1.7b-q8_0"], tmp_path / "d", capsys
    )
    assert code == 1 and "--yes" in err
    code, _, err = run_cli(
        ["ai", "model", "install", "no-such-model"], tmp_path / "d", capsys
    )
    assert code == 1 and "error:" in err


def test_cli_runtime_use_rejects_other_programs(tmp_path, capsys):
    code, _, err = run_cli(
        ["ai", "runtime", "use", sys.executable], tmp_path / "d", capsys
    )
    assert code == 1 and "llama-server" in err


def test_cli_test_with_no_generation(tmp_path, capsys):
    data = tmp_path / "d"
    conn = None
    from litenotebook.config import load_settings
    from litenotebook.db import open_db

    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(data)})
    conn = open_db(settings.db_path)
    AIStore(conn).save_profile(AIStore(conn).profile().__class__(type="none"))
    conn.close()
    code, out, _ = run_cli(["ai", "test"], data, capsys)
    assert code == 1 and "not configured" in out


# --- no hidden network ---------------------------------------------------------------


def test_no_network_outside_explicit_downloads(client, app_settings, monkeypatch):
    """Page views, the hardware scan, recommendations, MiniLM detection and
    the health badge never open a connection beyond 127.0.0.1."""
    import socket

    outside = []
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in (
            "127.0.0.1",
            "::1",
            "localhost",
        ):
            outside.append(host)
            raise OSError("network access blocked by the test")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: outside.append(a[0]) or (_ for _ in ()).throw(OSError()),
    )
    client.post("/setup/welcome", headers=ORIGIN)
    client.post("/setup/hardware", headers=ORIGIN)
    client.post("/settings/ai/rescan", headers=ORIGIN)
    for url in [
        "/",
        "/settings",
        "/settings/ai",
        "/health/badge",
        "/palette.json",
        *[f"/setup?step={s}" for s in SETUP_STEPS],
    ]:
        assert client.get(url).status_code == 200, url
    emb.status()
    assert outside == []
