"""Static configuration: filesystem paths and pipeline defaults.

Values come from ``LITENOTEBOOK_*`` environment variables or the defaults
below. User-editable LLM settings will later live in the SQLite ``settings``
table and override the LLM defaults here at runtime.

This module deliberately does not read ``.env`` files: the repository's
``.env`` belongs to the legacy Open Notebook stack.

Where data lives (``load_settings``), in order:

1. ``LITENOTEBOOK_DATA_DIR`` — an explicit custom folder (``%VAR%``/``$VAR``
   and ``~`` are expanded the way the running OS does);
2. a source checkout (``pyproject.toml`` next to ``src/litenotebook``) runs in
   **development mode**: ``<checkout>/data``, exactly as before Phase 11;
3. otherwise (an installed copy, or ``LITENOTEBOOK_MODE=user``) the platform
   default: ``~/.local/share/LiteNotebook`` (XDG) or
   ``%LOCALAPPDATA%\\LiteNotebook``.

Existing data is never moved between these locations automatically.
Config, cache and log folders follow the same rule
(``LITENOTEBOOK_CONFIG_DIR`` / ``LITENOTEBOOK_CACHE_DIR`` /
``LITENOTEBOOK_LOG_DIR`` override them); in development and custom mode they
sit inside the data folder. Platform log folders:
``%LOCALAPPDATA%\\LiteNotebook\\Logs`` and ``$XDG_STATE_HOME/LiteNotebook/logs``
(``~/.local/state``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODES = ("development", "user")
ORIGIN_CUSTOM = "custom (LITENOTEBOOK_DATA_DIR)"
ORIGIN_DEVELOPMENT = "development checkout"
ORIGIN_PLATFORM = "platform default"


def is_source_checkout(root: Path = PROJECT_ROOT) -> bool:
    if getattr(sys, "frozen", False):  # a packaged app is never "development"
        return False
    return (root / "pyproject.toml").is_file() and (
        root / "src" / "litenotebook"
    ).is_dir()


# Embeddings: fixed by the architecture. Qwen is for generation only.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
EMBED_MAX_TOKENS = 256  # all-MiniLM-L6-v2 max_seq_length; longer input is truncated

# Generation defaults (overridable via LITENOTEBOOK_* env vars or the settings page)
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B-GGUF:Q4_K_M"
DEFAULT_TOP_K = 8

DEFAULT_MAX_UPLOAD_MB = 100

ENV_PREFIX = "LITENOTEBOOK_"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class ConfigError(ValueError):
    """Invalid configuration value."""


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    host: str = "127.0.0.1"
    data_dir_origin: str = ORIGIN_CUSTOM
    config_dir: Path | None = None  # default: <data_dir>/config
    cache_dir: Path | None = None  # default: <data_dir>/cache
    log_dir: Path | None = None  # default: <data_dir>/logs
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

    # Uploads (web): PDFs larger than this are rejected while streaming.
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB

    # Backup / export archives (ZIP). Archives are hostile input: every limit
    # below is enforced while reading, not taken from the ZIP headers.
    archive_max_mb: int = 4096  # compressed archive (upload or file)
    archive_max_expanded_mb: int = 8192  # sum of all uncompressed entries
    archive_max_entry_mb: int = 2048  # any single entry (e.g. one PDF)
    archive_max_files: int = 20000

    # MiniLM runs in a child process that exits after this long without work
    # (its memory goes back to the OS); the next embedding restarts it.
    embed_worker_idle_s: int = 120
    embed_worker_start_timeout_s: int = 180

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
        if not 1 <= self.max_upload_mb <= 4096:
            raise ConfigError("max_upload_mb must be between 1 and 4096")
        for name in (
            "archive_max_mb",
            "archive_max_expanded_mb",
            "archive_max_entry_mb",
        ):
            if not 1 <= getattr(self, name) <= 1_048_576:
                raise ConfigError(f"{name} must be between 1 and 1048576")
        if not 10 <= self.archive_max_files <= 1_000_000:
            raise ConfigError("archive_max_files must be between 10 and 1000000")
        if not 1 <= self.embed_worker_idle_s <= 86400:
            raise ConfigError("embed_worker_idle_s must be between 1 and 86400")
        if not 10 <= self.embed_worker_start_timeout_s <= 3600:
            raise ConfigError(
                "embed_worker_start_timeout_s must be between 10 and 3600"
            )

    @property
    def config_path(self) -> Path:
        return (
            self.config_dir if self.config_dir is not None else self.data_dir / "config"
        )

    @property
    def cache_path(self) -> Path:
        return self.cache_dir if self.cache_dir is not None else self.data_dir / "cache"

    @property
    def log_path(self) -> Path:
        """Per-user log folder (startup, worker and runtime failures)."""
        return self.log_dir if self.log_dir is not None else self.data_dir / "logs"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

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

    def source_file(self, notebook_id: int, source_id: int) -> Path:
        """Storage path for an uploaded source: built from integer ids only."""
        return self.sources_dir(int(notebook_id)) / f"{int(source_id)}.pdf"

    @property
    def archive_tmp_dir(self) -> Path:
        """Archive uploads, exports in progress and import staging (same
        filesystem as the data, so staged files move by rename)."""
        return self.data_dir / "tmp" / "archives"

    @property
    def upload_tmp_dir(self) -> Path:
        """Partial uploads land here (same filesystem as sources, not tmpfs /tmp)."""
        return self.data_dir / "tmp" / "uploads"


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
        ("max_upload_mb", "MAX_UPLOAD_MB"),
        ("embed_worker_idle_s", "EMBED_WORKER_IDLE_SECONDS"),
        ("embed_worker_start_timeout_s", "EMBED_WORKER_START_TIMEOUT"),
        ("archive_max_mb", "ARCHIVE_MAX_MB"),
        ("archive_max_expanded_mb", "ARCHIVE_MAX_EXPANDED_MB"),
        ("archive_max_entry_mb", "ARCHIVE_MAX_ENTRY_MB"),
        ("archive_max_files", "ARCHIVE_MAX_FILES"),
    ):
        if (number := get_int(name)) is not None:
            overrides[key] = number

    def path(raw: str) -> Path:
        return Path(os.path.expandvars(raw)).expanduser().absolute()

    mode = get("MODE")
    if mode is not None and mode not in MODES:
        raise ConfigError(f"{ENV_PREFIX}MODE must be one of {', '.join(MODES)}")
    explicit = get("DATA_DIR")
    config_dir = cache_dir = log_dir = None
    if explicit:
        data_dir, origin = path(explicit), ORIGIN_CUSTOM
    elif mode == "user" or (mode is None and not is_source_checkout()):
        from litenotebook.platform_utils.system import app_dirs

        dirs = app_dirs(env)
        data_dir, origin = Path(dirs.data), ORIGIN_PLATFORM
        config_dir, cache_dir = Path(dirs.config), Path(dirs.cache)
        log_dir = Path(dirs.log) if dirs.log is not None else None
    else:
        data_dir, origin = PROJECT_ROOT / "data", ORIGIN_DEVELOPMENT
    if raw_config := get("CONFIG_DIR"):
        config_dir = path(raw_config)
    if raw_cache := get("CACHE_DIR"):
        cache_dir = path(raw_cache)
    if raw_log := get("LOG_DIR"):
        log_dir = path(raw_log)
    return Settings(
        data_dir=data_dir,
        data_dir_origin=origin,
        config_dir=config_dir,
        cache_dir=cache_dir,
        log_dir=log_dir,
        **overrides,  # type: ignore[arg-type]
    )
