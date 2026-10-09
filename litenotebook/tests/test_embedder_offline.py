"""MiniLM loads from the local cache only: no Hub requests, no implicit download."""

import os
import subprocess
import sys
import textwrap
import types

import pytest

from conftest import FakeEmbedder, make_pdf
from litenotebook.config import EMBED_DIM, EMBED_MODEL
from litenotebook.embed import embedder as emb
from litenotebook.embed.embedder import (
    EmbeddingModelUnavailableError,
    MiniLMEmbedder,
    local_model_dir,
)
from litenotebook.ingest.pipeline import IngestError, process_source, register_source
from litenotebook.sources import get_source

# Installed before anything else in the subprocesses below: every outbound
# connection attempt (DNS or TCP) is recorded and refused.
NETWORK_GUARD = """
import socket
ATTEMPTS = []
def _refuse(kind):
    def guard(*args, **kwargs):
        ATTEMPTS.append((kind, repr(args[:2])[:120]))
        raise OSError(f"network disabled by test ({kind})")
    return guard
socket.getaddrinfo = _refuse("getaddrinfo")
socket.create_connection = _refuse("create_connection")
socket.socket.connect = _refuse("connect")
socket.socket.connect_ex = _refuse("connect_ex")
"""


def _online_env(**extra):
    """The user's normal environment: no HF offline switches set."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    env.update(extra)
    return env


def _run(code, env):
    out = subprocess.run(
        [sys.executable, "-c", NETWORK_GUARD + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


@pytest.fixture(autouse=True)
def fresh_caches():
    local_model_dir.cache_clear()
    emb.get_tokenizer.cache_clear()
    yield
    local_model_dir.cache_clear()
    emb.get_tokenizer.cache_clear()


# --- fast unit tests (no model, no torch) -------------------------------------------


def test_model_lookup_is_local_only(monkeypatch, tmp_path):
    import huggingface_hub

    calls = []
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name in ("modules.json", "config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).write_text("{}")

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert local_model_dir() == snapshot
    assert calls == [(EMBED_MODEL, {"local_files_only": True})]


def test_missing_model_raises_actionable_error(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def not_cached(repo_id, **kwargs):
        assert kwargs.get("local_files_only") is True  # never a download
        raise LocalEntryNotFoundError("not in cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    with pytest.raises(EmbeddingModelUnavailableError) as exc:
        MiniLMEmbedder().load()
    message = str(exc.value)
    assert EMBED_MODEL in message and "not available locally" in message
    assert "does not download" in message and f"hf download {EMBED_MODEL}" in message
    assert "/" + "home" not in message  # no filesystem paths in UI-facing text
    with pytest.raises(EmbeddingModelUnavailableError):
        emb.count_tokens("tokenizer needs the local model too")


def test_incomplete_snapshot_is_reported(monkeypatch, tmp_path):
    import huggingface_hub

    (tmp_path / "modules.json").write_text("{}")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, **kw: str(tmp_path)
    )
    with pytest.raises(EmbeddingModelUnavailableError, match="model weights"):
        local_model_dir()


def test_sentence_transformer_gets_local_dir_and_local_files_only(
    monkeypatch, tmp_path
):
    import huggingface_hub

    for name in ("modules.json", "config.json", "tokenizer.json", "model.safetensors"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, **kw: str(tmp_path)
    )
    seen = {}

    class FakeST:
        max_seq_length = 256

        def __init__(self, name_or_path, **kwargs):
            seen["args"], seen["kwargs"] = name_or_path, kwargs

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeST),
    )
    e = MiniLMEmbedder()
    assert not e.loaded
    e.load()
    assert e.loaded
    assert seen["args"] == str(tmp_path)  # a directory, not a Hub repo id
    assert seen["kwargs"] == {"device": "cpu", "local_files_only": True}


def test_missing_model_fails_source_with_useful_message(
    conn, settings, notebook, store, tmp_path
):
    class Unavailable(FakeEmbedder):
        def encode(self, texts):
            raise EmbeddingModelUnavailableError()

    pdf = make_pdf(tmp_path / "a.pdf", ["Some text about pivots and rank."])
    src = register_source(conn, settings, notebook, pdf)
    with pytest.raises(IngestError, match="not available locally"):
        process_source(conn, settings, src.id, Unavailable(), store)
    failed = get_source(conn, notebook.id, src.id)
    assert failed.status == "failed" and "hf download" in failed.error
    assert failed.chunk_count == 0 and store.check(notebook.id).ok


# --- real model, real environment (subprocess) ---------------------------------------


@pytest.mark.model
def test_cached_minilm_loads_and_embeds_with_network_disabled():
    out = _run(
        """
        import numpy as np
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer
        from litenotebook.embed.embedder import MiniLMEmbedder, count_tokens
        texts = ["Gaussian elimination uses pivots.", "x" * 900, "Ünïcødé ∑ text"]
        # The Phase 1 tokenizer source, read straight from the cache. Not
        # hf_hub_download(local_files_only=True): since huggingface_hub 1.33
        # it still builds request headers, which refresh an "agent registry"
        # from huggingface.co once its 24-hour cache is stale.
        old = Tokenizer.from_file(try_to_load_from_cache(
            "sentence-transformers/all-MiniLM-L6-v2", "tokenizer.json"))
        old.no_truncation()
        old.no_padding()
        n = all(count_tokens(t) == len(old.encode(t).ids) for t in texts)
        e = MiniLMEmbedder()
        v = e.encode(["Gaussian elimination uses pivots.", "Rank equals pivots."])
        import torch
        print("TOKENS", n)
        print("SHAPE", v.shape, "NORM", round(float(np.linalg.norm(v[0])), 4))
        print("CUDA", torch.version.cuda, torch.cuda.is_available())
        print("ATTEMPTS", len(ATTEMPTS), ATTEMPTS[:3])
        """,
        _online_env(),
    )
    assert "ATTEMPTS 0 []" in out
    assert f"SHAPE (2, {EMBED_DIM}) NORM 1.0" in out
    assert "TOKENS True" in out  # identical token counts to the Phase 1 tokenizer
    assert "CUDA None False" in out


@pytest.mark.model
def test_missing_local_model_never_tries_to_download(tmp_path):
    empty = tmp_path / "hf-empty"
    out = _run(
        """
        from litenotebook.embed.embedder import (
            EmbeddingModelUnavailableError, MiniLMEmbedder)
        try:
            MiniLMEmbedder().encode(["anything"])
        except EmbeddingModelUnavailableError as e:
            print("ERROR", e)
        print("ATTEMPTS", len(ATTEMPTS), ATTEMPTS[:3])
        """,
        _online_env(HF_HOME=str(empty), HF_HUB_CACHE=str(empty / "hub")),
    )
    assert "ERROR The embedding model" in out and "not available locally" in out
    assert "ATTEMPTS 0 []" in out
    assert not (empty / "hub").exists() or not any(
        (empty / "hub").rglob("*.safetensors")
    )
