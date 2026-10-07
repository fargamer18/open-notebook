"""Phase 0 guards: configuration invariants and the shape of the environment."""

import re
from importlib import metadata
from pathlib import Path

import pytest

from litenotebook.config import (
    EMBED_DIM,
    EMBED_MAX_TOKENS,
    EMBED_MODEL,
    PROJECT_ROOT,
    ConfigError,
    load_settings,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "litenotebook"

FORBIDDEN_DISTRIBUTIONS = {
    "langchain",
    "langchain-core",
    "langchain-community",
    "langgraph",
    "llama-index",
    "llama-index-core",
    "surrealdb",
    "surreal-commands",
    "redis",
    "celery",
    "psycopg",
    "psycopg2",
    "psycopg2-binary",
    "elasticsearch",
    "docker",
}


def _installed() -> set[str]:
    return {
        re.sub(r"[-_.]+", "-", (d.metadata["Name"] or "")).lower()
        for d in metadata.distributions()
    }


# --- configuration -----------------------------------------------------------


def test_defaults(settings):
    assert settings.host == "127.0.0.1"
    assert settings.port == 5056
    assert settings.llm_base_url == "http://127.0.0.1:8080/v1"
    assert settings.chunk_overlap_tokens < settings.chunk_target_tokens
    assert settings.chunk_target_tokens <= EMBED_MAX_TOKENS
    assert settings.db_path.name == "litenotebook.sqlite3"
    assert settings.faiss_index_path(3) == settings.data_dir / "notebooks/3/index.faiss"


def test_default_data_dir_is_inside_project():
    assert load_settings({}).data_dir == PROJECT_ROOT / "data"


def test_embeddings_use_minilm_not_qwen():
    assert EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert EMBED_DIM == 384
    assert EMBED_MAX_TOKENS == 256
    assert "qwen" not in EMBED_MODEL.lower()


def test_env_overrides(tmp_path):
    s = load_settings(
        {
            "LITENOTEBOOK_DATA_DIR": str(tmp_path),
            "LITENOTEBOOK_PORT": "6000",
            "LITENOTEBOOK_LLM_BASE_URL": "http://localhost:9000/v1",
            "LITENOTEBOOK_CHUNK_TOKENS": "180",
        }
    )
    assert (s.data_dir, s.port, s.llm_base_url) == (
        tmp_path,
        6000,
        "http://localhost:9000/v1",
    )
    assert s.chunk_target_tokens == 180


@pytest.mark.parametrize(
    "env",
    [
        {"LITENOTEBOOK_HOST": "0.0.0.0"},
        {"LITENOTEBOOK_CHUNK_TOKENS": "300"},  # exceeds MiniLM's 256 limit
        {
            "LITENOTEBOOK_CHUNK_TOKENS": "100",
            "LITENOTEBOOK_CHUNK_OVERLAP_TOKENS": "100",
        },
        {"LITENOTEBOOK_LLM_BASE_URL": "file:///etc/passwd"},
        {"LITENOTEBOOK_PORT": "abc"},
    ],
)
def test_invalid_config_rejected(env, tmp_path):
    with pytest.raises(ConfigError):
        load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path), **env})


# --- environment -------------------------------------------------------------


def test_no_forbidden_runtime_packages():
    assert FORBIDDEN_DISTRIBUTIONS.isdisjoint(_installed())


def test_torch_is_cpu_only():
    import torch

    assert torch.version.cuda is None
    assert "+cpu" in torch.__version__
    gpu_packages = sorted(
        n for n in _installed() if n.startswith(("nvidia-", "triton", "cuda-"))
    )
    assert gpu_packages == []


def test_faiss_idmap_flat_ip_roundtrip(tmp_path):
    import faiss
    import numpy as np

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(EMBED_DIM))
    vectors = np.eye(3, EMBED_DIM, dtype="float32")
    index.add_with_ids(vectors, np.array([101, 205, 999], dtype="int64"))
    index.remove_ids(np.array([205], dtype="int64"))

    path = tmp_path / "index.faiss"
    faiss.write_index(index, str(path))
    loaded = faiss.read_index(str(path))
    _, ids = loaded.search(vectors[2:3], 1)
    assert loaded.ntotal == 2
    assert ids[0][0] == 999


def test_core_dependencies_import():
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    import jinja2  # noqa: F401
    import markdown_it  # noqa: F401
    import nh3  # noqa: F401
    import pymupdf
    import sentence_transformers  # noqa: F401

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "hello page one")
    assert "hello page one" in doc[0].get_text()


def test_package_does_not_depend_on_legacy_or_open_notebook():
    pattern = re.compile(
        r"^\s*(from|import)\s+(legacy|open_notebook|api|commands)\b", re.M
    )
    offenders = [p for p in SRC.rglob("*.py") if pattern.search(p.read_text())]
    assert offenders == []
