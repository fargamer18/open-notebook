"""One implementation of local-AI setup, shared by the web UI and the CLI.

Everything that changes state is an explicit call: scanning hardware,
installing a runtime/model/MiniLM, choosing an existing file, starting,
stopping, testing. Nothing here runs on page loads except cheap status reads
(no network, no model loads, no runtime starts).

How Chat reaches a model (``generation_endpoint``):

* ``managed_llama`` / ``external_llama`` — the llama-server LiteNotebook
  starts, at ``http://127.0.0.1:<port>/v1`` with the model's alias;
* ``external_openai`` — the server configured under Settings (URL, model,
  optional API key), exactly as before Phase 12;
* ``unset`` (never set up) — the same Settings values as before, so existing
  installations keep working; failures then point to setup;
* ``none`` — the user chose no generation: Chat says so and links to setup.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from litenotebook.ai import catalog as cat
from litenotebook.ai import embeddings as emb
from litenotebook.ai.check import BENCH_PROMPT, check_generation
from litenotebook.ai.download import Downloader, Progress
from litenotebook.ai.hardware import HardwareProfile, scan_hardware
from litenotebook.ai.models import (
    ModelError,
    import_model,
    install_model,
    model_status,
    models_dir,
    read_gguf,
    remove_model,
)
from litenotebook.ai.planner import (
    Fit,
    LaunchPlan,
    available_backends,
    fallback_plans,
    plan_launch,
    recommend,
)
from litenotebook.ai.runtime_manager import (
    LaunchSpec,
    RuntimeManager,
    RuntimeStartError,
    spec_for,
)
from litenotebook.ai.runtimes import (
    RuntimeError_,
    install_runtime,
    list_installed,
    probe,
    runtime_root,
)
from litenotebook.ai.store import AIStore, GenerationProfile, ModelRecord
from litenotebook.config import Settings
from litenotebook.platform_utils.filesystem import is_within
from litenotebook.platform_utils.system import os_name

NOT_CONFIGURED = "Local AI is not configured."


class SetupError(RuntimeError):
    """A setup step failed; the message says what and what to do."""


class FallbackError(SetupError):
    """Starting failed; ``changes`` lists the safer settings that were tried."""

    def __init__(self, message: str, changes: list[str]) -> None:
        super().__init__(message)
        self.changes = changes


@dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: str | None
    managed: bool


@dataclass
class TestOutcome:
    ok: bool
    message: str
    load_s: float | None = None
    ttft_s: float | None = None
    tokens_per_s: float | None = None
    rss_bytes: int | None = None
    reply: str = ""
    changes: list[str] | None = None
    config: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalAI:
    def __init__(
        self,
        settings: Settings,
        runtime: RuntimeManager,
        *,
        downloader_factory: Callable[[], Downloader] = Downloader,
        minilm_command: list[str] | None = None,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.downloader_factory = downloader_factory
        self.minilm_command = minilm_command
        self._start_lock = threading.Lock()
        self._autostart_lock = threading.Lock()

    # --- read-only views -------------------------------------------------------------

    def hardware(
        self, conn: sqlite3.Connection, *, rescan: bool = False
    ) -> HardwareProfile:
        store = AIStore(conn)
        cached = store.hardware()
        if cached and not rescan:
            with contextlib.suppress(TypeError, KeyError):
                return HardwareProfile.from_dict(cached)
        hw = scan_hardware(self.settings.data_dir)
        store.save_hardware(hw.as_dict())
        return hw

    def runtime_options(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        hw = self.hardware(conn)
        installed = {r.id for r in list_installed(self.settings)}
        gpu = hw.best_gpu
        out = []
        for entry in cat.runtimes_for(hw.platform, hw.architecture):
            ok, why = True, ""
            if entry.backend == "cuda":
                need = str(entry.requires.get("min_nvidia_driver") or "0")
                driver = gpu.driver_version if gpu and gpu.vendor == "nvidia" else None
                if gpu is None or gpu.vendor != "nvidia":
                    ok, why = False, "needs an NVIDIA GPU"
                elif driver is None or _version_tuple(driver) < _version_tuple(need):
                    ok, why = False, f"needs NVIDIA driver {need} or newer"
            out.append(
                {
                    "entry": entry,
                    "suitable": ok,
                    "why": why,
                    "installed": entry.id in installed,
                }
            )
        return out

    def recommendations(self, conn: sqlite3.Connection) -> list[Fit]:
        hw = self.hardware(conn)
        store = AIStore(conn)
        profile = store.profile()
        runtime_backends = ["cpu"]
        if profile.backend == "cuda" or any(
            o["suitable"] and o["entry"].backend == "cuda"
            for o in self.runtime_options(conn)
        ):
            runtime_backends.append("cuda")
        installed = frozenset(m.catalog_id for m in store.models() if m.catalog_id)
        return recommend(
            cat.load_models(),
            hw,
            available_backends(hw, runtime_backends),
            installed_ids=installed,
            minilm=emb.status()["installed"],
        )

    def overview(self, conn: sqlite3.Connection) -> dict[str, Any]:
        store = AIStore(conn)
        profile = store.profile()
        models = store.models()
        active = next((m for m in models if m.id == profile.model_id), None)
        return {
            "setup": store.setup(),
            "profile": profile,
            "runtime": self.runtime.status(),
            "runtimes": list_installed(self.settings),
            "models": [(m, model_status(m)) for m in models],
            "active_model": active,
            "embeddings": {**emb.status(), **store.embeddings()},
            "hardware": store.hardware(),
            "models_dir": models_dir(self.settings, store),
            "components": components(profile, active, store, self.speech_status()),
            "speech": self.speech_status(),
        }

    # --- speech (optional, Phase 14) -------------------------------------------------

    def speech_status(self) -> dict[str, Any]:
        from litenotebook.ai import speech

        return speech.status(self.settings)

    def install_speech_engine(
        self,
        *,
        progress: Callable[[str, Progress | None], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Any:
        from litenotebook.ai import speech

        entry = speech.engine_entry()
        if entry is None:
            raise SetupError(
                "There is no speech engine build for this computer in the catalog"
            )
        downloader = self.downloader_factory()
        try:
            return speech.install_engine(
                self.settings,
                entry,
                downloader=downloader,
                progress=progress,
                cancel=cancel,
            )
        except speech.SpeechError as e:
            raise SetupError(str(e)) from e
        finally:
            downloader.close()

    def install_voice(
        self,
        voice_id: str,
        *,
        progress: Callable[[str, Progress | None], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> Any:
        from litenotebook.ai import speech

        try:
            entry = speech.catalog().voice(voice_id)
        except cat.CatalogError as e:
            raise SetupError("Unknown voice") from e
        downloader = self.downloader_factory()
        try:
            return speech.install_voice(
                self.settings,
                entry,
                downloader=downloader,
                progress=progress,
                cancel=cancel,
            )
        except speech.SpeechError as e:
            raise SetupError(str(e)) from e
        finally:
            downloader.close()

    # --- generation runtime choices --------------------------------------------------

    def install_runtime(
        self,
        conn: sqlite3.Connection,
        runtime_id: str,
        *,
        progress: Callable[[str, Progress | None], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> GenerationProfile:
        entry = cat.runtime(runtime_id)
        downloader = self.downloader_factory()
        try:
            installed = install_runtime(
                self.settings,
                entry,
                downloader=downloader,
                progress=progress,
                cancel=cancel,
            )
        except RuntimeError_ as e:
            raise SetupError(str(e)) from e
        finally:
            downloader.close()
        store = AIStore(conn)
        profile = store.profile()
        profile.type = "managed_llama"
        profile.runtime_id, profile.runtime_version = installed.id, installed.version
        profile.backend = installed.backend
        profile.executable = (
            Path(installed.executable)
            .relative_to(runtime_root(self.settings))
            .as_posix()
        )
        store.save_profile(profile)
        store.mark("generation")
        self._replan(conn)
        return store.profile()

    def use_existing_runtime(
        self, conn: sqlite3.Connection, path: str
    ) -> GenerationProfile:
        result = probe(Path(path).expanduser())
        if not result.ok:
            raise SetupError(f"That llama-server cannot be used: {result.error}")
        store = AIStore(conn)
        profile = store.profile()
        profile.type = "external_llama"
        profile.executable = str(Path(result.executable).resolve())
        profile.runtime_id = None
        profile.runtime_version = f"{result.version} (build {result.build})"
        profile.backend = result.backend
        store.save_profile(profile)
        store.mark("generation")
        self._replan(conn)
        return store.profile()

    def use_external_server(self, conn: sqlite3.Connection) -> GenerationProfile:
        store = AIStore(conn)
        profile = store.profile()
        profile.type = "external_openai"
        store.save_profile(profile)
        store.mark("generation")
        store.mark("model", skipped=True)
        return profile

    def skip_generation(self, conn: sqlite3.Connection) -> GenerationProfile:
        store = AIStore(conn)
        profile = store.profile()
        profile.type = "none"
        store.save_profile(profile)
        store.mark("generation", skipped=True)
        store.mark("model", skipped=True)
        store.mark("test", skipped=True)
        return profile

    # --- models ----------------------------------------------------------------------

    def install_model(
        self,
        conn: sqlite3.Connection,
        model_id: str,
        *,
        progress: Callable[[Progress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> ModelRecord:
        entry = cat.model(model_id)
        downloader = self.downloader_factory()
        try:
            record = install_model(
                self.settings,
                AIStore(conn),
                entry,
                downloader=downloader,
                progress=progress,
                cancel=cancel,
            )
        finally:
            downloader.close()
        self.select_model(conn, record.id)
        return record

    def import_model(
        self,
        conn: sqlite3.Connection,
        path: str,
        *,
        copy: bool = False,
        cancel: threading.Event | None = None,
    ) -> ModelRecord:
        try:
            record = import_model(
                self.settings, AIStore(conn), Path(path), copy=copy, cancel=cancel
            )
        except ModelError as e:
            raise SetupError(str(e)) from e
        self.select_model(conn, record.id)
        return record

    def select_model(
        self, conn: sqlite3.Connection, model_id: str
    ) -> GenerationProfile:
        store = AIStore(conn)
        if store.model(model_id) is None:
            raise SetupError("That model is not installed")
        profile = store.profile()
        if self.runtime.running() and profile.model_id != model_id:
            self.runtime.stop()
        profile.model_id = model_id
        store.save_profile(profile)
        store.mark("model")
        self._replan(conn)
        return store.profile()

    def remove_model(self, conn: sqlite3.Connection, model_id: str) -> bool:
        store = AIStore(conn)
        profile = store.profile()
        if profile.model_id == model_id:
            self.runtime.stop()
            profile.model_id = None
            store.save_profile(profile)
        try:
            return remove_model(self.settings, store, model_id)
        except ModelError as e:
            raise SetupError(str(e)) from e

    # --- launch configuration --------------------------------------------------------

    def _entry_for(self, record: ModelRecord) -> cat.ModelEntry:
        if record.catalog_id:
            with contextlib.suppress(cat.CatalogError):
                return cat.model(record.catalog_id)
        arch = record.architecture or {
            "layers": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "max_context": 4096,
        }
        return cat.ModelEntry(
            id=record.id,
            display_name=record.display_name,
            family="other",
            publisher="",
            parameter_class="?",
            size_class="balanced",
            quantization=record.quantization or "?",
            filename=Path(record.path).name,
            repository="",
            revision="",
            url="",
            sha256="0" * 64,
            size=record.size,
            architecture=arch,
            recommended_contexts=(4096,),
            capabilities=(),
            alias=record.alias or record.id,
            runtime_args=tuple(record.runtime_args),
            model_page="",
            license="",
            license_url="",
            min_runtime_version="",
            notes="",
        )

    def plan(
        self, conn: sqlite3.Connection, *, live: bool = False
    ) -> LaunchPlan | None:
        store = AIStore(conn)
        profile = store.profile()
        record = store.model(profile.model_id or "")
        if record is None:
            return None
        hw = self.hardware(conn, rescan=live)
        backend = "cuda" if profile.backend == "cuda" else "cpu"
        return plan_launch(
            self._entry_for(record),
            hw,
            backend=backend,
            live_vram=live,
            minilm=emb.status()["installed"],
        )

    def _replan(self, conn: sqlite3.Connection) -> None:
        plan = self.plan(conn)
        if plan is None:
            return
        store = AIStore(conn)
        profile = store.profile()
        _apply(profile, plan)
        profile.note = ""
        store.save_profile(profile)

    def _executable(self, profile: GenerationProfile) -> str:
        if profile.type == "managed_llama":
            root = runtime_root(self.settings)
            exe = (root / (profile.executable or "")).resolve()
            if not profile.executable or not exe.is_file() or not is_within(exe, root):
                raise SetupError(
                    "The installed llama.cpp runtime is missing; repair it in "
                    "Settings → Local AI"
                )
            return str(exe)
        exe = Path(profile.executable or "")
        if not exe.is_file():
            raise SetupError(
                "The llama-server program configured for Local AI is missing"
            )
        return str(exe)

    def launch_spec(self, conn: sqlite3.Connection) -> tuple[LaunchSpec, ModelRecord]:
        store = AIStore(conn)
        profile = store.profile()
        if not profile.launches_runtime:
            raise SetupError("Local AI is not set up to run llama.cpp on this computer")
        record = store.model(profile.model_id or "")
        if record is None:
            raise SetupError("Choose a generation model first")
        path = Path(record.path)
        if not path.is_file():
            raise SetupError(f"The model file for {record.display_name} is missing")
        try:
            read_gguf(path)  # never launch anything that is not a GGUF model
        except ModelError as e:
            raise SetupError(str(e)) from e
        spec = spec_for(
            profile,
            self._executable(profile),
            str(path),
            record.alias or record.id,
            record.runtime_args,
        )
        return spec, record

    # --- runtime lifecycle -----------------------------------------------------------

    def start(
        self, conn: sqlite3.Connection, on_progress: Callable[[str], None] | None = None
    ) -> dict[str, Any]:
        spec, _ = self.launch_spec(conn)
        with self._start_lock:
            try:
                status = self.runtime.start(spec, on_progress=on_progress)
            except RuntimeStartError as e:
                raise SetupError(str(e)) from e
        if status.get("port") and status["port"] != spec.port:
            store = AIStore(conn)
            profile = store.profile()
            profile.port = int(status["port"])
            store.save_profile(profile)
        return status

    def start_with_fallback(
        self,
        conn: sqlite3.Connection,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Start with a plan for the memory free *now* (another program may be
        using the GPU), then step down the fallback ladder after an
        out-of-memory exit. The settings that worked are saved."""
        say = progress or (lambda _m: None)
        store = AIStore(conn)
        profile = store.profile()
        record = store.model(profile.model_id or "")
        if record is None:
            raise SetupError("Choose a generation model first")
        plan = self.plan(conn, live=True) or LaunchPlan(
            profile.context,
            profile.gpu_layers,
            profile.kv_type,
            profile.flash_attn,
            profile.threads,
            profile.parallel,
            "cpu",
            0,
            0,
        )
        attempts = [(plan, "")] + fallback_plans(self._entry_for(record), plan)
        changes: list[str] = []
        last_error = ""
        for attempt, change in attempts:
            profile = store.profile()
            _apply(profile, attempt)
            store.save_profile(profile)
            if change:
                changes.append(change)
                say(f"Retrying: {change}…")
            say("Starting the AI runtime…")
            try:
                status = self.start(conn, on_progress=say)
            except SetupError as e:
                last_error = str(e)
                if "did not fit in memory" not in last_error:
                    raise FallbackError(last_error, changes) from e
                continue
            if changes:
                profile = store.profile()
                profile.note = (
                    "The first configuration did not fit in memory. LiteNotebook "
                    "retried and " + "; then ".join(changes) + "."
                )
                store.save_profile(profile)
            return status, changes
        raise FallbackError(
            f"{last_error}. Even the safest settings failed; choose a smaller model.",
            changes,
        )

    def stop(self) -> str | None:
        return self.runtime.stop()

    def ensure_running(self, conn: sqlite3.Connection) -> None:
        """For features that need generation (Chat, later quizzes/podcasts)."""
        profile = AIStore(conn).profile()
        if not profile.launches_runtime or self.runtime.running():
            return
        if not profile.autostart:
            raise SetupError("Local AI is stopped. Start it in Settings → Local AI.")
        with self._autostart_lock:  # two questions at once start it once
            if not self.runtime.running():
                self.start_with_fallback(conn)

    def generation_endpoint(
        self, conn: sqlite3.Connection, legacy: Any
    ) -> Endpoint | None:
        """``legacy`` is the effective LLM settings (SettingsService)."""
        store = AIStore(conn)
        profile = store.profile()
        if profile.type == "none":
            return None
        if profile.launches_runtime:
            record = store.model(profile.model_id or "")
            if record is None:
                return None
            port = self.runtime.status().get("port", profile.port)
            return Endpoint(
                f"http://127.0.0.1:{port}/v1", record.alias or record.id, None, True
            )
        return Endpoint(
            legacy.llm_base_url, legacy.llm_model, legacy.llm_api_key or None, False
        )

    # --- tests -----------------------------------------------------------------------

    def test(
        self,
        conn: sqlite3.Connection,
        legacy: Any,
        *,
        progress: Callable[[str], None] | None = None,
        benchmark: bool = False,
    ) -> TestOutcome:
        say = progress or (lambda _m: None)
        store = AIStore(conn)
        profile = store.profile()
        if profile.type == "none":
            return TestOutcome(False, NOT_CONFIGURED)
        changes: list[str] = []
        load_s = None
        if profile.launches_runtime:
            if not self.runtime.running():
                try:
                    status, changes = self.start_with_fallback(conn, progress=say)
                except SetupError as e:
                    changes = e.changes if isinstance(e, FallbackError) else []
                    return TestOutcome(False, str(e), changes=changes)
                load_s = status.get("load_s")
                profile = store.profile()
            else:
                load_s = self.runtime.status().get("load_s")
        endpoint = self.generation_endpoint(conn, legacy)
        if endpoint is None:
            return TestOutcome(False, NOT_CONFIGURED)
        say("Asking the model a test question…")
        result = check_generation(
            endpoint.base_url,
            endpoint.model,
            api_key=endpoint.api_key,
            prompt=BENCH_PROMPT if benchmark else "Reply with exactly one word: ready",
            max_tokens=160 if benchmark else 24,
        )
        rss = self.runtime.status().get("rss_bytes") if endpoint.managed else None
        if (
            result.ok
            and endpoint.managed
            and endpoint.model not in result.models_listed
        ):
            result.ok, result.error = (
                False,
                "The server is not serving the selected model",
            )
        config = {
            "context": profile.context,
            "gpu_layers": profile.gpu_layers,
            "backend": profile.backend,
            "kv_type": profile.kv_type,
            "type": profile.type,
        }
        summary = {
            "ok": result.ok,
            "at": time.time(),
            "load_s": load_s,
            "ttft_s": result.ttft_s,
            "tokens_per_s": result.tokens_per_s,
            "rss_bytes": rss,
            "config": config,
            "error": result.error,
        }
        if benchmark:
            profile.last_benchmark = summary
        else:
            profile.last_test = summary
        store.save_profile(profile)
        if result.ok and not benchmark:
            store.mark("test")
        message = (
            "Your local AI answered."
            if result.ok
            else result.error or "The test failed"
        )
        return TestOutcome(
            result.ok,
            message,
            load_s,
            result.ttft_s,
            result.tokens_per_s,
            rss,
            result.reply,
            changes,
            config,
        )

    # --- embeddings ------------------------------------------------------------------

    def install_embeddings(
        self,
        conn: sqlite3.Connection,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        try:
            result = emb.install(
                progress=progress, cancel=cancel, command=self.minilm_command
            )
        except emb.EmbeddingSetupError as e:
            raise SetupError(str(e)) from e
        store = AIStore(conn)
        store.save_embeddings({"state": "installed", "installed_at": time.time()})
        store.mark("embeddings")
        return result

    def skip_embeddings(self, conn: sqlite3.Connection) -> None:
        store = AIStore(conn)
        store.save_embeddings({"state": "skipped"})
        store.mark("embeddings", skipped=True)


def _apply(profile: GenerationProfile, plan: LaunchPlan) -> None:
    profile.context, profile.gpu_layers = plan.context, plan.gpu_layers
    profile.kv_type, profile.flash_attn = plan.kv_type, plan.flash_attn
    profile.threads, profile.parallel = plan.threads, plan.parallel


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for p in text.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def components(
    profile: GenerationProfile,
    active: ModelRecord | None,
    store: AIStore,
    speech: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The AI components shown in Settings, each with its own catalog."""
    embeddings = store.embeddings()
    generation_ready = (
        profile.type == "external_openai"
        or (profile.launches_runtime and active is not None)
        or profile.type == "unset"
    )
    return [
        {
            "id": "generation",
            "label": "Generation",
            "optional": False,
            "state": "off"
            if profile.type == "none"
            else ("ready" if generation_ready else "incomplete"),
            "used_by": ["Chat", "Quizzes", "Audio Overview scripts"],
        },
        {
            "id": "embeddings",
            "label": "Embeddings",
            "optional": True,
            "state": "skipped" if embeddings.get("state") == "skipped" else "check",
            "used_by": ["Hybrid search", "Chat retrieval", "Quiz and audio focus"],
        },
        {
            "id": "speech",
            "label": "Speech",
            "optional": True,
            "state": "ready"
            if speech and speech.get("ready")
            else ("incomplete" if speech and speech.get("installed") else "off"),
            "used_by": ["Audio Overviews (audio only; transcripts work without it)"],
        },
    ]


def platform_name() -> str:
    return {"linux": "Linux", "windows": "Windows", "macos": "macOS"}.get(
        os_name(), os_name()
    )
