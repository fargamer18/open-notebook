"""User-editable settings stored in SQLite.

Precedence per key: ``LITENOTEBOOK_*`` environment variable > SQLite > default.
Environment values are never written to the database. The LLM API key is
never logged, rendered, or returned by ``public()``.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

import httpx

from litenotebook.config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    ENV_PREFIX,
)
from litenotebook.llm.client import LLMClient, LLMError, LLMResponseError

LLM_BASE_URL = "llm.base_url"
LLM_MODEL = "llm.model"
LLM_API_KEY = "llm.api_key"
RAG_TOP_K = "rag.top_k"

ENV_VARS = {
    LLM_BASE_URL: ENV_PREFIX + "LLM_BASE_URL",
    LLM_MODEL: ENV_PREFIX + "LLM_MODEL",
    LLM_API_KEY: ENV_PREFIX + "LLM_API_KEY",
    RAG_TOP_K: ENV_PREFIX + "TOP_K",
}
DEFAULTS = {
    LLM_BASE_URL: DEFAULT_LLM_BASE_URL,
    LLM_MODEL: DEFAULT_LLM_MODEL,
    LLM_API_KEY: "",
    RAG_TOP_K: str(DEFAULT_TOP_K),
}
TOP_K_RANGE = (1, 20)
CONNECT_TIMEOUT_S = 5.0

Source = Literal["env", "db", "default"]


class SettingsError(ValueError):
    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def validate_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value or len(value) > 500:
        raise ValueError("Enter the server URL, e.g. http://127.0.0.1:8080/v1")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Must be an http:// or https:// URL with a host")
    if parts.username or parts.password:
        raise ValueError("Do not put credentials in the URL; use the API key field")
    if parts.query or parts.fragment:
        raise ValueError("The URL must not contain a query string or fragment")
    return value


def validate_model(value: str) -> str:
    value = value.strip()
    if len(value) > 200:
        raise ValueError("Model name must be at most 200 characters")
    if any(not ch.isprintable() for ch in value):
        raise ValueError("Model name contains invalid characters")
    return value  # empty = let the server use its loaded model


def validate_top_k(value: str | int) -> int:
    try:
        number = int(str(value).strip())
    except ValueError as e:
        raise ValueError("Must be a whole number") from e
    low, high = TOP_K_RANGE
    if not low <= number <= high:
        raise ValueError(f"Must be between {low} and {high}")
    return number


def validate_api_key(value: str) -> str:
    value = value.strip()
    if len(value) > 512:
        raise ValueError("API key is too long")
    if any(ch.isspace() or not ch.isprintable() for ch in value):
        raise ValueError("API key must not contain spaces or control characters")
    return value


def _validate_top_k_str(value: str) -> str:
    return str(validate_top_k(value))


_VALIDATORS: dict[str, Callable[[str], str]] = {
    LLM_BASE_URL: validate_base_url,
    LLM_MODEL: validate_model,
    LLM_API_KEY: validate_api_key,
    RAG_TOP_K: _validate_top_k_str,
}


@dataclass(frozen=True)
class AppSettings:
    llm_base_url: str
    llm_model: str
    llm_api_key: str = field(repr=False)  # never in repr/logs
    rag_top_k: int
    sources: dict[str, Source]

    def public(self) -> dict[str, object]:
        """Safe for templates and JSON: the API key itself is never included."""
        return {
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "rag_top_k": self.rag_top_k,
            "api_key_configured": bool(self.llm_api_key),
            "sources": dict(self.sources),
            "env_vars": dict(ENV_VARS),
        }

    def llm_client(
        self,
        *,
        timeout_s: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> LLMClient:
        return LLMClient(
            self.llm_base_url,
            self.llm_model,
            api_key=self.llm_api_key or None,
            timeout_s=timeout_s,
            transport=transport,
        )


class SettingsService:
    def __init__(
        self, conn: sqlite3.Connection, env: Mapping[str, str] | None = None
    ) -> None:
        self.conn = conn
        self.env = os.environ if env is None else env

    def _stored(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows if r["key"] in DEFAULTS}

    def effective(self) -> AppSettings:
        stored = self._stored()
        values: dict[str, str] = {}
        sources: dict[str, Source] = {}
        for key, default in DEFAULTS.items():
            env_value = self.env.get(ENV_VARS[key], "").strip()
            if env_value:
                try:
                    values[key] = _VALIDATORS[key](env_value)
                except ValueError as e:
                    raise SettingsError({key: f"{ENV_VARS[key]}: {e}"}) from e
                sources[key] = "env"
            elif key in stored:
                values[key], sources[key] = stored[key], "db"
            else:
                values[key], sources[key] = default, "default"
        return AppSettings(
            llm_base_url=values[LLM_BASE_URL],
            llm_model=values[LLM_MODEL],
            llm_api_key=values[LLM_API_KEY],
            rag_top_k=int(values[RAG_TOP_K]),
            sources=sources,
        )

    def update(
        self,
        *,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        rag_top_k: str | int | None = None,
        llm_api_key: str | None = None,
        clear_api_key: bool = False,
    ) -> AppSettings:
        """Validate everything first, then write. ``llm_api_key`` of None or ""
        keeps the stored key (the form never receives it back)."""
        raw: dict[str, str | int | None] = {
            LLM_BASE_URL: llm_base_url,
            LLM_MODEL: llm_model,
            RAG_TOP_K: rag_top_k,
            LLM_API_KEY: llm_api_key or None,
        }
        clean: dict[str, str] = {}
        errors: dict[str, str] = {}
        for key, value in raw.items():
            if value is None:
                continue
            try:
                clean[key] = _VALIDATORS[key](str(value))
            except ValueError as e:
                errors[key] = str(e)
        if errors:
            raise SettingsError(errors)
        with self.conn:
            for key, value in clean.items():
                self.conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                    (key, value),
                )
            if clear_api_key and LLM_API_KEY not in clean:
                self.conn.execute("DELETE FROM settings WHERE key = ?", (LLM_API_KEY,))
        return self.effective()


@dataclass(frozen=True)
class ConnectionReport:
    ok: bool
    reachable: bool
    models: list[str]
    model: str
    model_found: bool | None  # None when no model is selected
    message: str
    elapsed_ms: float


def check_connection(
    settings: AppSettings,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = CONNECT_TIMEOUT_S,
) -> ConnectionReport:
    """Reachability + /models + selected-model check. Never generates text."""
    started = time.perf_counter()

    def report(**kw: object) -> ConnectionReport:
        elapsed = (time.perf_counter() - started) * 1000
        return ConnectionReport(model=settings.llm_model, elapsed_ms=elapsed, **kw)  # type: ignore[arg-type]

    try:
        with settings.llm_client(timeout_s=timeout_s, transport=transport) as llm:
            models = llm.list_models()
    except LLMError as e:
        return report(
            ok=False,
            reachable=isinstance(e, LLMResponseError),
            models=[],
            model_found=None,
            message=str(e),
        )

    if not settings.llm_model:
        return report(
            ok=True,
            reachable=True,
            models=models,
            model_found=None,
            message=f"Connected — {len(models)} model(s) available; no model selected "
            "(the server's loaded model will be used)",
        )
    if settings.llm_model in models:
        return report(
            ok=True,
            reachable=True,
            models=models,
            model_found=True,
            message=f"Connected — {settings.llm_model} available",
        )
    available = ", ".join(models) or "none"
    return report(
        ok=False,
        reachable=True,
        models=models,
        model_found=False,
        message=f"Connected, but model {settings.llm_model!r} was not found. "
        f"Available: {available}",
    )
