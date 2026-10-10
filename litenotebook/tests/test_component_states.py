"""The core UI launches in every optional-component state (Phase 16).

States: nothing installed; generation only; embeddings only; both; speech
installed but broken; a model whose file was removed outside the app; a
voice folder removed. In each, every main page renders (200, no traceback),
nothing is downloaded, no AI process is started by page views, and the
pages say what is missing instead of failing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ai_fixtures import fake_gguf
from audio_fixtures import install_fake_engine, install_fake_voices
from conftest import FakeEmbedder, make_pdf
from litenotebook.ai import embeddings as emb
from litenotebook.ai.store import AIStore
from litenotebook.config import load_settings
from litenotebook.db import connect, open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.notebooks import create_notebook
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
PAGES = (
    "/",
    "/setup",
    "/settings",
    "/settings/ai",
    "/settings/diagnostics",
    "/backup",
    "/notebooks/1",
    "/notebooks/1?tab=sources",
    "/notebooks/1/chat",
    "/notebooks/1/notes",
    "/notebooks/1/search?q=Velt",
    "/notebooks/1/memory",
    "/notebooks/1/study",
    "/notebooks/1/study/quizzes/new",
    "/notebooks/1/study/audio/new",
    "/notebooks/1/sources/1/read",
    "/health/badge",
)


def _installed(flag: bool):  # type: ignore[no-untyped-def]
    def cached(e=None):  # type: ignore[no-untyped-def]
        e = e or emb.entry()
        return {
            f.path: (Path("/nonexistent") if not flag else Path(__file__))
            for f in e.files
        }

    return cached


@pytest.mark.parametrize(
    "state",
    [
        "nothing",
        "generation_only",
        "embeddings_only",
        "generation_and_embeddings",
        "speech_broken",
        "model_file_removed",
        "voice_removed",
    ],
)
def test_core_ui_launches_in_every_component_state(tmp_path, monkeypatch, state):
    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    conn = open_db(settings.db_path)
    nb = create_notebook(conn, "States")
    ingest_pdf(
        conn,
        settings,
        nb,
        make_pdf(tmp_path / "v.pdf", ["Velt river rises."]),
        FakeEmbedder(),
        FaissStore(conn, settings),
    )
    conn.close()

    has_embeddings = state in ("embeddings_only", "generation_and_embeddings")
    monkeypatch.setattr(emb, "cached_files", _installed(has_embeddings))

    if state in ("generation_only", "generation_and_embeddings", "model_file_removed"):
        c = connect(settings.db_path)
        store = AIStore(c)
        profile = store.profile()
        profile.type, profile.executable = (
            "external_llama",
            str(tmp_path / "llama-server"),
        )
        store.save_profile(profile)
        from litenotebook.ai.runtime_manager import RuntimeManager
        from litenotebook.ai.service import LocalAI

        model = fake_gguf(tmp_path / "models" / "tiny.gguf")
        LocalAI(settings, RuntimeManager(settings)).import_model(c, str(model))
        c.close()
        if state == "model_file_removed":
            model.unlink()
    if state in ("speech_broken", "voice_removed"):
        install_fake_engine(settings)
        install_fake_voices(settings)
        from litenotebook.ai.speech import speech_root

        root = speech_root(settings)
        if state == "speech_broken":
            for exe in (root / "engine").rglob("sherpa-onnx-offline-tts*"):
                exe.write_bytes(b"\x7fELF-broken")
        else:
            shutil.rmtree(root / "voices" / "voice-b")

    network: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        network.append(str(request.url))
        raise httpx.ConnectError("offline", request=request)

    started: list[object] = []
    real_popen = subprocess.Popen

    def popen(*a, **k):  # type: ignore[no-untyped-def]
        started.append(a[0] if a else k.get("args"))
        return real_popen(*a, **k)

    monkeypatch.setattr(subprocess, "Popen", popen)
    app = create_app(
        settings,
        env={},
        llm_transport=httpx.MockTransport(transport),
        embedder_factory=FakeEmbedder,
    )
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as client:
        for path in PAGES:
            r = client.get(path, headers={"Accept": "text/html"})
            assert r.status_code == 200, (state, path, r.status_code)
            assert "Traceback" not in r.text and "Something went wrong" not in r.text, (
                path
            )
        ai_page = client.get("/settings/ai").text
        if state == "model_file_removed":
            assert "missing" in ai_page.lower()
    # page views never download anything or start llama/speech/worker processes
    assert not [u for u in network if "huggingface" in u or "github" in u], network
    assert not [
        s
        for s in started
        if "llama" in str(s) or "sherpa" in str(s) or "embed" in str(s)
    ], started
