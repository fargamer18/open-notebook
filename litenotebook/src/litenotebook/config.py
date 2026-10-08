"""Static configuration: filesystem paths and pipeline defaults.

Values come from ``LITENOTEBOOK_*`` environment variables or the defaults
below. User-editable LLM settings will later live in the SQLite ``settings``
table and override the LLM defaults here at runtime.

This module deliberately does not read ``.env`` files: the repository's
``.env`` belongs to the legacy Open Notebook stack.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Embeddings: fixed by the architecture. Qwen is for generation only.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_MAX_TOKENS = 256  # all-MiniLM-L6-v2 max_seq_length; longer input is truncated

# Generation defaults (overridable via LITENOTEBOOK_* env vars or the settings page)
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B-GGUF:Q4_K_M"
DEFAULT_TOP_K = 8

ENV_PREFIX = "LITENOTEBOOK_"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class ConfigError(ValueError):
    """Invalid configuration value."""


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 5056

    # Chunking, measured in MiniLM wordpiece tokens.
    chunk_target_tokens: int = 200
    chunk_overlap_tokens: int = 40

    # Retrieval
    retrieval_candidates: int = 30  # per retriever (FTS5 and FAISS) before fusion
    retrieval_top_k: int = DEFAULT_TOP_K
    rrf_k: int = 60
    dedup_jaccard_threshold: float = 0.8

    # Generation: OpenAI-compatible llama.cpp server
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_timeout_s: float = 180.0

    def __post_init__(self) -> None:
        if self.host not in LOOPBACK_HOSTS:
            raise ConfigError(
                f"host must be a loopback address, got {self.host!r}; "
                "LiteNotebook has no authentication"
            )
        if not 0 < self.chunk_target_tokens <= EMBED_MAX_TOKENS:
            raise ConfigError(
                f"chunk_target_tokens must be in 1..{EMBED_MAX_TOKENS}, "
                f"got {self.chunk_target_tokens}"
            )
        if not 0 <= self.chunk_overlap_tokens < self.chunk_target_tokens:
            raise ConfigError(
                "chunk_overlap_tokens must be >= 0 and < chunk_target_tokens"
            )
        if not self.llm_base_url.startswith(("http://", "https://")):
            raise ConfigError("llm_base_url must be an http(s) URL")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "litenotebook.sqlite3"

    @property
    def notebooks_dir(self) -> Path:
        return self.data_dir / "notebooks"

    def notebook_dir(self, notebook_id: int) -> Path:
        return self.notebooks_dir / str(notebook_id)

    def faiss_index_path(self, notebook_id: int) -> Path:
        return self.notebook_dir(notebook_id) / "index.faiss"

    def sources_dir(self, notebook_id: int) -> Path:
        return self.notebook_dir(notebook_id) / "sources"


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings from ``LITENOTEBOOK_*`` variables, falling back to defaults."""
    env = os.environ if env is None else env

    def get(name: str) -> str | None:
        value = env.get(ENV_PREFIX + name)
        return value.strip() if value and value.strip() else None

    def get_int(name: str) -> int | None:
        raw = get(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError as e:
            raise ConfigError(f"{ENV_PREFIX}{name} must be an integer") from e

    overrides: dict[str, object] = {}
    for key, name in (
        ("host", "HOST"),
        ("llm_base_url", "LLM_BASE_URL"),
        ("llm_model", "LLM_MODEL"),
    ):
        if (value := get(name)) is not None:
            overrides[key] = value
    for key, name in (
        ("port", "PORT"),
        ("chunk_target_tokens", "CHUNK_TOKENS"),
        ("chunk_overlap_tokens", "CHUNK_OVERLAP_TOKENS"),
        ("retrieval_top_k", "TOP_K"),
    ):
        if (number := get_int(name)) is not None:
            overrides[key] = number

    data_dir = Path(get("DATA_DIR") or PROJECT_ROOT / "data").expanduser()
    return Settings(data_dir=data_dir, **overrides)  # type: ignore[arg-type]
