"""Persistent AI setup state: wizard progress, runtime profile, models.

Stored as JSON values under ``ai.*`` keys of the existing ``settings``
table — no schema change (the database stays at version 8). These keys are
machine-specific (paths, ports, hardware), so full backups deliberately do not
carry them (``archive.format.BACKUP_SETTINGS`` is an allowlist). Secrets never
live here: an external server's API key stays in ``llm.api_key``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

SETUP_STEPS = (
    "welcome",
    "hardware",
    "embeddings",
    "generation",
    "model",
    "test",
    "ready",
)
RUNTIME_TYPES = (
    "unset",  # never configured: the pre-Phase-12 LLM settings apply as before
    "managed_llama",  # llama.cpp installed by LiteNotebook
    "external_llama",  # the user's own llama-server binary, started by LiteNotebook
    "external_openai",  # an already-running OpenAI-compatible server
    "none",  # the user chose not to use generation
)
KEY_SETUP, KEY_PROFILE, KEY_MODELS, KEY_EMBEDDINGS, KEY_HARDWARE, KEY_MODELS_DIR = (
    "ai.setup",
    "ai.generation",
    "ai.models",
    "ai.embeddings",
    "ai.hardware",
    "ai.models_dir",
)


@dataclass
class SetupState:
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    completed: bool = False
    updated_at: float = 0.0

    def next_step(self) -> str:
        for step in SETUP_STEPS:
            if step not in self.done and step not in self.skipped:
                return step
        return "ready"


@dataclass
class GenerationProfile:
    type: str = "unset"
    executable: str | None = None  # managed: relative to the runtime root
    runtime_id: str | None = None
    runtime_version: str | None = None
    backend: str = "cpu"
    model_id: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    context: int = 4096
    gpu_layers: int = 0
    threads: int = 4
    parallel: int = 1
    flash_attn: str = "on"
    kv_type: str = "q8_0"
    autostart: bool = True
    last_test: dict[str, Any] | None = None
    last_benchmark: dict[str, Any] | None = None
    note: str = ""  # e.g. what a fallback changed

    @property
    def launches_runtime(self) -> bool:
        return self.type in ("managed_llama", "external_llama")


@dataclass
class ModelRecord:
    id: str
    display_name: str
    path: str
    owned: bool  # True only for files LiteNotebook downloaded/copied itself
    size: int
    catalog_id: str | None = None
    sha256: str | None = None
    verified: bool = False  # checksum matched a catalog entry
    quantization: str | None = None
    alias: str | None = None
    architecture: dict[str, int] | None = None
    runtime_args: list[str] = field(default_factory=list)
    added_at: float = 0.0


class AIStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def _get(self, key: str) -> Any:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def _put(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    # setup wizard
    def setup(self) -> SetupState:
        data = self._get(KEY_SETUP) or {}
        known = {k: data[k] for k in SetupState.__dataclass_fields__ if k in data}
        return SetupState(**known)

    def save_setup(self, state: SetupState) -> None:
        state.updated_at = time.time()
        self._put(KEY_SETUP, asdict(state))

    def mark(self, step: str, *, skipped: bool = False) -> SetupState:
        state = self.setup()
        target = state.skipped if skipped else state.done
        other = state.done if skipped else state.skipped
        if step in other:
            other.remove(step)
        if step not in target:
            target.append(step)
        if step == "ready" or all(
            s in state.done or s in state.skipped for s in SETUP_STEPS
        ):
            state.completed = True
        self.save_setup(state)
        return state

    def reset_setup(self) -> None:
        """Re-run the wizard; never touches notebooks or installed files."""
        self.save_setup(SetupState())

    # generation runtime profile
    def profile(self) -> GenerationProfile:
        data = self._get(KEY_PROFILE) or {}
        known = {
            k: data[k] for k in GenerationProfile.__dataclass_fields__ if k in data
        }
        profile = GenerationProfile(**known)
        if profile.type not in RUNTIME_TYPES:
            profile.type = "unset"
        return profile

    def save_profile(self, profile: GenerationProfile) -> None:
        self._put(KEY_PROFILE, asdict(profile))

    # models
    def models(self) -> list[ModelRecord]:
        out = []
        for d in self._get(KEY_MODELS) or []:
            if isinstance(d, dict):
                known = {k: d[k] for k in ModelRecord.__dataclass_fields__ if k in d}
                try:
                    out.append(ModelRecord(**known))
                except TypeError:
                    continue
        return out

    def save_models(self, records: list[ModelRecord]) -> None:
        self._put(KEY_MODELS, [asdict(r) for r in records])

    def model(self, model_id: str) -> ModelRecord | None:
        return next((m for m in self.models() if m.id == model_id), None)

    # embeddings, hardware, model folder
    def embeddings(self) -> dict[str, Any]:
        return self._get(KEY_EMBEDDINGS) or {}

    def save_embeddings(self, value: dict[str, Any]) -> None:
        self._put(KEY_EMBEDDINGS, value)

    def hardware(self) -> dict[str, Any] | None:
        value = self._get(KEY_HARDWARE)
        return value if isinstance(value, dict) else None

    def save_hardware(self, value: dict[str, Any]) -> None:
        self._put(KEY_HARDWARE, value)

    def models_dir(self) -> str | None:
        value = self._get(KEY_MODELS_DIR)
        return value if isinstance(value, str) and value else None

    def save_models_dir(self, value: str | None) -> None:
        self._put(KEY_MODELS_DIR, value or "")
