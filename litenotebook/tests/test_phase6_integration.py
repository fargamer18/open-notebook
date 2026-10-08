"""Phase 6 integration: the web app, ingestion, chat and CLI on the worker."""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import make_pdf
from litenotebook.config import load_settings
from litenotebook.db import available_migrations, connect, migrate, open_db
from litenotebook.embed.embedder import count_tokens
from litenotebook.embed.worker_client import WorkerEmbedder
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.chunker import chunk_pages
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.notebooks import create_notebook
from litenotebook.sources import get_source
from litenotebook.web.app import create_app
from test_chat_web import FakeLlama, ask
from test_embed_worker import pids, process_gone

BASE = "http://127.0.0.1:5056"
ORIGIN = {"Origin": BASE}
HX = {**ORIGIN, "HX-Request": "true"}
HEAVY = ("torch", "sentence_transformers", "transformers")


def fake_env(tmp_path, mode="", delay=0.0):
    return {
        "LITENOTEBOOK_EMBED_WORKER_BACKEND": "test-fake",
        "LITENOTEBOOK_EMBED_WORKER_TEST_DIR": str(tmp_path),
        "LITENOTEBOOK_EMBED_WORKER_TEST_MODE": mode,
        "LITENOTEBOOK_EMBED_WORKER_TEST_DELAY": str(delay),
    }


@pytest.fixture
def settings_(tmp_path):
    return load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})


def app_with_worker(settings_, tmp_path, llama=None, mode="", delay=0.0, idle=60):
    holder = {}

    def factory():
        holder["w"] = WorkerEmbedder(
            idle_seconds=idle, env=fake_env(tmp_path, mode, delay)
        )
        return holder["w"]

    app = create_app(
        settings_,
        env={},
        llm_transport=httpx.MockTransport(llama or FakeLlama()),
        embedder_factory=factory,
    )
    return app, holder


def upload_and_wait(c, app, nb, pdf, timeout=30):
    c.post(
        f"/notebooks/{nb}/sources",
        files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
        headers=HX,
    )
    assert app.state.worker.wait_idle(timeout)


# --- import isolation -----------------------------------------------------------------


def test_parent_modules_never_import_the_ml_stack():
    code = (
        "import sys\n"
        "import litenotebook, litenotebook.web.app, litenotebook.cli\n"
        "import litenotebook.retrieval.hybrid, litenotebook.ingest.pipeline\n"
        "import litenotebook.services.chat, litenotebook.services.ingestion\n"
        "import litenotebook.memory.search, litenotebook.memory.graph\n"
        "import litenotebook.embed.worker_client, litenotebook.embed.protocol\n"
        "import litenotebook.embed.worker, litenotebook.index.faiss_store\n"
        "from litenotebook.services.embedding import worker_embedder_factory\n"
        "worker_embedder_factory()  # creating the client must not import either\n"
        f"print('HEAVY', [m for m in {HEAVY!r} if m in sys.modules])\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "HEAVY []" in out.stdout


def test_browsing_never_starts_the_worker(settings_, tmp_path):
    app, holder = app_with_worker(settings_, tmp_path)
    conn = connect(settings_.db_path)
    nb = create_notebook(conn, "n").id
    conn.close()
    with TestClient(app, base_url=BASE) as c:
        for url in (
            "/",
            f"/notebooks/{nb}",
            f"/notebooks/{nb}?tab=sources",
            f"/notebooks/{nb}/chat",
            f"/notebooks/{nb}/memory",
            f"/notebooks/{nb}/memory?q=anything",
            "/settings",
            "/healthz",
            "/health/badge",
        ):
            assert c.get(url).status_code == 200, url
        assert c.get("/healthz").json()["embedding_worker"] == {"status": "stopped"}
        assert "Sleeping" in c.get("/settings").text
    assert holder == {} and pids(tmp_path) == []


# --- ingestion -----------------------------------------------------------------------


def test_ingestion_through_worker_with_loading_stage(settings_, tmp_path):
    app, holder = app_with_worker(settings_, tmp_path, mode="slow_load", delay=1.5)
    conn = connect(settings_.db_path)
    nb = create_notebook(conn, "n").id
    conn.close()
    pdf = make_pdf(tmp_path / "doc.pdf", ["Pivots and rank.", "Quillport harbour."])
    with TestClient(app, base_url=BASE) as c:
        c.post(
            f"/notebooks/{nb}/sources",
            files={"file": ("doc.pdf", pdf.read_bytes(), "application/pdf")},
            headers=HX,
        )
        seen = set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            row = c.get(f"/notebooks/{nb}/sources/1/status", headers=HX).text
            if "Loading embedding model" in row:
                seen.add("loading")
            if "pill-ready" in row:
                break
            time.sleep(0.05)
        assert "loading" in seen
        health = c.get("/healthz").json()["embedding_worker"]
        assert health["status"] == "ready" and health["pid"] == holder["w"].pid
        conn = connect(settings_.db_path)
        src = get_source(conn, nb, 1)
        assert (
            src.status == "ready"
            and src.chunk_count == 2
            and "model_load" in src.timings
        )
        assert FaissStore(conn, settings_).check(nb).ok
        conn.close()
        pid = holder["w"].pid
    assert process_gone(pid, 10), "server shutdown must reap the worker"


def test_ingestion_failure_then_retry_on_fresh_worker(settings_, tmp_path):
    app, holder = app_with_worker(settings_, tmp_path, mode="crash_on_embed")
    conn = connect(settings_.db_path)
    nb = create_notebook(conn, "n").id
    conn.close()
    pdf = make_pdf(tmp_path / "doc.pdf", ["Pivots and rank."])
    with TestClient(app, base_url=BASE) as c:
        upload_and_wait(c, app, nb, pdf)
        conn = connect(settings_.db_path)
        failed = get_source(conn, nb, 1)
        assert failed.status == "failed" and "embedding worker stopped" in failed.error
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        assert FaissStore(conn, settings_).check(nb).ok
        conn.close()
        holder["w"].extra_env["LITENOTEBOOK_EMBED_WORKER_TEST_MODE"] = ""
        c.post(f"/notebooks/{nb}/sources/1/retry", headers=HX)
        assert app.state.worker.wait_idle(30)
        conn = connect(settings_.db_path)
        assert get_source(conn, nb, 1).status == "ready"
        conn.close()
    assert all(process_gone(pid, 10) for pid in pids(tmp_path))


def test_idle_worker_exits_and_restarts_for_next_job(settings_, tmp_path):
    app, holder = app_with_worker(settings_, tmp_path, idle=0.6)
    conn = connect(settings_.db_path)
    nb = create_notebook(conn, "n").id
    conn.close()
    with TestClient(app, base_url=BASE) as c:
        upload_and_wait(c, app, nb, make_pdf(tmp_path / "a.pdf", ["First document."]))
        first = holder["w"].pid or pids(tmp_path)[0]
        assert process_gone(first, 10)
        assert c.get("/healthz").json()["embedding_worker"]["status"] == "stopped"
        assert "Sleeping" in c.get("/settings").text
        upload_and_wait(c, app, nb, make_pdf(tmp_path / "b.pdf", ["Second document."]))
        conn = connect(settings_.db_path)
        assert [get_source(conn, nb, i).status for i in (1, 2)] == ["ready", "ready"]
        conn.close()
    assert len(pids(tmp_path)) == 2


# --- chat -----------------------------------------------------------------------------


def test_chat_shows_loading_then_retrieves_through_worker(settings_, tmp_path):
    llama = FakeLlama(chunks=["Quillport [S1]."])
    app, holder = app_with_worker(settings_, tmp_path, llama, idle=0.6)
    conn = connect(settings_.db_path)
    nb = create_notebook(conn, "n").id
    conn.close()
    with TestClient(app, base_url=BASE) as c:
        upload_and_wait(
            c, app, nb, make_pdf(tmp_path / "z.pdf", ["Quillport harbour."])
        )
        assert process_gone(pids(tmp_path)[0], 10)  # worker went to sleep
        r, events = ask(c, nb, "Where is the harbour?")
        stages = [d["stage"] for e, d in events if e == "status"]
        assert stages == ["loading_model", "retrieving", "generating"]
        assert events[-1][0] == "done" and len(llama.requests) == 1
        assert "Quillport harbour." in llama.last_prompt


def test_chat_embedding_failure_never_calls_the_llm(settings_, tmp_path):
    llama = FakeLlama()
    conn = open_db(settings_.db_path)
    nb = create_notebook(conn, "n")
    from conftest import FakeEmbedder

    ingest_pdf(
        conn,
        settings_,
        nb,
        make_pdf(tmp_path / "z.pdf", ["Quillport."]),
        FakeEmbedder(),
        FaissStore(conn, settings_),
    )
    conn.close()
    app, holder = app_with_worker(settings_, tmp_path, llama, mode="crash_on_embed")
    with TestClient(app, base_url=BASE) as c:
        r, events = ask(c, nb.id, "Where is Quillport?")
        assert r.status_code == 200 and events[-1][0] == "error"
        assert "embedding worker stopped unexpectedly" in events[-1][1]["message"]
        html = c.get(
            f"/notebooks/{nb.id}/chat/messages/{events[0][1]['message_id']}", headers=HX
        ).text
        assert "data-retry-url" in html and "pill-failed" in html
        assert c.get(f"/notebooks/{nb.id}/chat").status_code == 200
    assert llama.requests == []


# --- CLI ------------------------------------------------------------------------------


def test_cli_uses_the_worker_and_leaves_no_orphan(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", ["Gaussian elimination and pivots."])
    env = {**__import__("os").environ, **fake_env(tmp_path)}
    base = [sys.executable, "-m", "litenotebook.cli", "--data-dir", str(tmp_path / "d")]
    for args in (
        ["notebook", "create", "demo"],
        ["ingest", "--notebook", "demo", str(pdf)],
        ["search", "--notebook", "demo", "pivots"],
        ["index", "check", "--notebook", "demo"],
    ):
        out = subprocess.run(
            base + args, capture_output=True, text=True, env=env, timeout=120
        )
        assert out.returncode == 0, (args, out.stderr)
    assert len(pids(tmp_path)) == 2  # ingest + search each used one worker
    assert all(process_gone(pid, 10) for pid in pids(tmp_path))


# --- regressions ---------------------------------------------------------------


def test_tokenizer_counts_and_chunks_unchanged_from_phase1():
    samples = [
        "Gaussian elimination uses pivots.",
        "x" * 900,
        "Ünïcødé ∑ text — eigenvalues λ₁, λ₂",
        "",
        "Hello, world! " * 40,
    ]
    assert [count_tokens(s) for s in samples] == [11, 3, 18, 2, 162]
    pages = [
        (
            1,
            " ".join(
                f"Paragraph {i} discusses matrices, determinants and the history "
                f"of algebra in considerable detail."
                for i in range(60)
            ),
        ),
        (2, "Short page about eigenvalues and eigenvectors."),
    ]
    chunks = chunk_pages(pages, target_tokens=200, overlap_tokens=40)
    sig = [(c.page_no, c.ord, c.char_start, c.char_end, c.n_tokens) for c in chunks]
    assert len(chunks) == 8
    assert (
        hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:16] == "5e1ff87400d0f50b"
    )


def test_migration_0006_preserves_sources_and_allows_loading_stage(
    settings, tmp_path, monkeypatch
):
    from conftest import FakeEmbedder
    from litenotebook import db

    partial = tmp_path / "migrations_5"
    partial.mkdir()
    for _v, path in available_migrations()[:5]:
        shutil.copy(path, partial / path.name)
    conn = connect(settings.db_path)
    with monkeypatch.context() as m:
        m.setattr(db, "MIGRATIONS_DIR", partial)
        assert migrate(conn) == 5
    nb = create_notebook(conn, "Maths")
    store = FaissStore(conn, settings)
    ingest_pdf(
        conn,
        settings,
        nb,
        make_pdf(tmp_path / "m.pdf", ["Pivots."]),
        FakeEmbedder(),
        store,
    )
    before = dict(conn.execute("SELECT * FROM sources").fetchone())
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert migrate(conn) == len(available_migrations()) >= 6
    assert dict(conn.execute("SELECT * FROM sources").fetchone()) == before
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == chunks
    assert (
        store.check(nb.id).ok
        and conn.execute("PRAGMA foreign_key_check").fetchall() == []
    )
    conn.execute("UPDATE sources SET stage = 'loading_model'")  # now a valid stage
    conn.close()


@pytest.mark.model
def test_real_minilm_worker_matches_in_process_semantics():
    code = """
import sys, json
import numpy as np
from litenotebook.embed.worker_client import WorkerEmbedder
texts = ["Gaussian elimination uses pivots.", "The capital of Zorbania is Quillport."]
w = WorkerEmbedder()
v = w.encode(texts)
q = w.encode([texts[0]])
pid, info = w.pid, w.info
w.close()
np.save(sys.argv[1], np.concatenate([v, q]))
heavy = [m for m in ("torch", "sentence_transformers") if m in sys.modules]
print(json.dumps({"pid": pid, "cuda": info.get("torch_cuda"),
                  "device": info.get("device"), "parent_heavy": heavy}))
"""
    out_file = Path(__import__("tempfile").mkdtemp()) / "v.npy"
    out = subprocess.run(
        [sys.executable, "-c", code, str(out_file)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    assert report["parent_heavy"] == [] and report["cuda"] is None
    assert report["device"] == "cpu" and process_gone(report["pid"], 10)
    from litenotebook.embed.embedder import MiniLMEmbedder

    worker = np.load(out_file)
    local = MiniLMEmbedder().encode(
        ["Gaussian elimination uses pivots.", "The capital of Zorbania is Quillport."]
    )
    np.testing.assert_allclose(worker[:2], local, atol=1e-5)
    np.testing.assert_allclose(worker[2], local[0], atol=1e-5)
    assert worker.dtype == np.float32 and worker.shape == (3, 384)
