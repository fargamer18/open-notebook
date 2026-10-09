"""Start, watch and stop the llama-server LiteNotebook launches.

* Only the configured executable is started (validated server-side), with an
  argument list (no shell), bound to 127.0.0.1, through
  ``platform_utils.processes`` (own process group; Windows Job Object).
* Readiness: poll ``/health`` (503 while loading, 200 when ready) with a
  bounded timeout; an early exit is reported with the reason found in the log
  (out of memory, port in use, missing file, crash).
* Port: if the configured port is busy, it is reused only when it is our own
  running process; otherwise a free local port is chosen. Nothing that
  LiteNotebook did not start is ever stopped.
* A CPU-only plan hides GPUs from the process (``CUDA_VISIBLE_DEVICES=""``),
  so a CUDA build uses no VRAM at all.
* Logs go to ``<data>/logs/llama-server.log`` (size-capped); the command line
  is logged without secrets (there are none: local server, no API key).
* Nothing starts on its own: only ``start()``/``ensure_running()`` calls —
  from Chat (when autostart is on), Settings, the wizard or the CLI.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import httpx

from litenotebook.ai.store import GenerationProfile
from litenotebook.config import Settings
from litenotebook.platform_utils.processes import (
    OwnedProcess,
    orphan_guard,
    spawn,
    stop,
)
from litenotebook.platform_utils.supervise import MARKER
from litenotebook.platform_utils.system import process_rss_bytes

log = logging.getLogger(__name__)
LOG_LIMIT = 5 * 1024 * 1024
OOM_MARKERS = (
    "out of memory",
    "cudamalloc failed",
    "failed to allocate",
    "unable to allocate",
    "not enough memory",
    "alloc_buffer: failed",
    "cuda error: out of memory",
)


class RuntimeStartError(RuntimeError):
    def __init__(self, kind: str, message: str, tail: list[str]) -> None:
        super().__init__(message)
        # out_of_memory | port | model | crash | timeout | executable | cancelled
        self.kind = kind
        self.tail = tail


@dataclass
class LaunchSpec:
    executable: str
    model_path: str
    alias: str
    port: int
    context: int
    gpu_layers: int
    threads: int
    parallel: int
    flash_attn: str
    kv_type: str
    runtime_args: tuple[str, ...] = ()

    def argv(self) -> list[str]:
        return [
            self.executable,
            "-m",
            self.model_path,
            "--alias",
            self.alias,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "-c",
            str(self.context),
            "-ngl",
            str(self.gpu_layers),
            "-np",
            str(self.parallel),
            "-t",
            str(self.threads),
            "-fa",
            self.flash_attn,
            "-ctk",
            self.kv_type,
            "-ctv",
            self.kv_type,
            *self.runtime_args,
        ]


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def classify(tail: list[str]) -> tuple[str, str]:
    text = "\n".join(tail).lower()
    if any(m in text for m in OOM_MARKERS):
        return "out_of_memory", "The model did not fit in memory with these settings"
    if "address already in use" in text or "couldn't bind" in text:
        return "port", "The port is already in use"
    if "failed to load model" in text or "error loading model" in text:
        return "model", "llama.cpp could not load the model file"
    return "crash", "llama-server stopped while starting"


class RuntimeManager:
    def __init__(
        self,
        settings: Settings,
        *,
        launcher: list[str] | None = None,  # tests: e.g. [python, fake_server.py]
        ready_timeout_s: float = 300.0,
        stop_grace_s: float = 10.0,
    ) -> None:
        self.settings = settings
        self.launcher = launcher
        self.ready_timeout_s = ready_timeout_s
        self.stop_grace_s = stop_grace_s
        self._lock = threading.RLock()
        self._owned: OwnedProcess | None = None
        self._spec: LaunchSpec | None = None
        self._state = "stopped"
        self._started_at = 0.0
        self._load_s: float | None = None
        self._last_error: str | None = None
        self._tail: collections.deque[str] = collections.deque(maxlen=200)
        self._sink: IO[str] | None = None
        self._abort = threading.Event()  # stop() during a start
        self._child_pid: int | None = None  # llama-server under the POSIX guard
        self.log_path = settings.data_dir / "logs" / "llama-server.log"

    # --- status (never starts anything) ----------------------------------------------

    def running(self) -> bool:
        owned = self._owned
        return (
            owned is not None and owned.proc.poll() is None and self._state == "running"
        )

    def status(self) -> dict[str, Any]:
        # A snapshot without ``_lock``: start() holds it while the model loads
        # (minutes), and status must answer meanwhile (badge, Settings).
        owned, spec = self._owned, self._spec
        alive = owned is not None and owned.proc.poll() is None
        if self._state == "running" and not alive:
            self._state = "failed"
            self._last_error = "llama-server exited unexpectedly"
        out: dict[str, Any] = {"state": self._state}
        if spec is not None:
            out.update(
                port=spec.port,
                alias=spec.alias,
                context=spec.context,
                gpu_layers=spec.gpu_layers,
            )
        if alive and owned is not None:
            pid = self._child_pid or owned.pid  # the llama-server itself
            out["pid"] = pid
            out["uptime_s"] = round(time.monotonic() - self._started_at, 1)
            rss = process_rss_bytes(pid)
            if rss:
                out["rss_bytes"] = rss
        if self._load_s is not None:
            out["load_s"] = self._load_s
        if self._last_error:
            out["last_error"] = self._last_error
        return out

    def base_url(self) -> str | None:
        spec = self._spec
        return f"http://127.0.0.1:{spec.port}/v1" if spec and self.running() else None

    def tail(self, n: int = 40) -> list[str]:
        return list(self._tail)[-n:]

    # --- start / stop ----------------------------------------------------------------

    def _pipe(self, stream: IO[bytes] | None, sink: IO[str]) -> None:
        if stream is None:
            return
        with contextlib.suppress(OSError, ValueError):
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if line.startswith(MARKER):
                    with contextlib.suppress(ValueError):
                        self._child_pid = int(line[len(MARKER) :])
                    continue
                self._tail.append(line)
                with contextlib.suppress(OSError, ValueError):
                    if sink.tell() < LOG_LIMIT:
                        sink.write(line + "\n")
                        sink.flush()

    def start(
        self, spec: LaunchSpec, *, on_progress: Callable[[str], None] | None = None
    ) -> dict[str, Any]:
        say = on_progress or (lambda _m: None)
        with self._lock:
            self._abort.clear()
            if self.running() and self._spec == spec:
                return self.status()
            self._stop_locked()
            exe = Path(spec.executable)
            if self.launcher is None and not exe.is_file():
                raise RuntimeStartError(
                    "executable", "The llama-server program was not found", []
                )
            if not Path(spec.model_path).is_file():
                raise RuntimeStartError("model", "The model file was not found", [])
            if port_open(spec.port):
                new_port = free_port()
                say(f"Port {spec.port} is used by another program; using {new_port}")
                spec = LaunchSpec(**{**spec.__dict__, "port": new_port})
            argv = spec.argv()
            if self.launcher is not None:
                argv = [*self.launcher, *argv[1:]]
            argv = orphan_guard(argv)
            env = dict(os.environ)
            if spec.gpu_layers == 0:
                env["CUDA_VISIBLE_DEVICES"] = (
                    ""  # CPU plan: keep a CUDA build off the GPU
                )
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            sink = open(self.log_path, "w", encoding="utf-8")  # noqa: SIM115 - closed on stop
            sink.write("$ " + " ".join(argv) + "\n")
            self._tail.clear()
            log.info("Starting llama-server: %s", " ".join(argv))
            try:
                owned = spawn(argv, env=env, tree=True)
            except OSError as e:
                sink.close()
                raise RuntimeStartError(
                    "executable",
                    f"llama-server could not be started ({e.strerror or e})",
                    [],
                ) from e
            self._owned, self._spec, self._state = owned, spec, "starting"
            self._child_pid = None
            self._sink = sink
            self._started_at = time.monotonic()
            self._load_s = None
            self._last_error = None
            for stream in (owned.stdout, owned.stderr):
                threading.Thread(
                    target=self._pipe, args=(stream, sink), daemon=True
                ).start()
            say("Loading the model…")
            deadline = time.monotonic() + self.ready_timeout_s
            url = f"http://127.0.0.1:{spec.port}/health"
            with httpx.Client(timeout=2.0) as client:
                while True:
                    if self._abort.is_set():
                        self._stop_locked()
                        self._last_error = "Start cancelled"
                        raise RuntimeStartError("cancelled", "Start cancelled", [])
                    if owned.proc.poll() is not None:
                        time.sleep(0.2)  # let the log readers catch the last lines
                        tail = self.tail(60)
                        kind, message = classify(tail)
                        self._stop_locked()
                        self._state, self._last_error = "failed", message
                        raise RuntimeStartError(kind, message, tail)
                    with contextlib.suppress(httpx.HTTPError):
                        if client.get(url).status_code == 200:
                            break
                    if time.monotonic() > deadline:
                        tail = self.tail(60)
                        self._stop_locked()
                        self._state = "failed"
                        self._last_error = "llama-server did not become ready in time"
                        raise RuntimeStartError("timeout", self._last_error, tail)
                    time.sleep(0.25)
            self._state = "running"
            self._load_s = round(time.monotonic() - self._started_at, 2)
            log.info(
                "llama-server ready on port %s in %.1fs (pid %s)",
                spec.port,
                self._load_s,
                owned.pid,
            )
            return self.status()

    def _stop_locked(self) -> str | None:
        owned, self._owned = self._owned, None
        if owned is None:
            self._state = "stopped"
            return None
        self._state = "stopping"
        outcome = stop(owned, graceful=owned.proc.terminate, grace_s=self.stop_grace_s)
        sink, self._sink = self._sink, None
        if sink is not None:
            with contextlib.suppress(OSError, ValueError):
                sink.close()
        self._state = "stopped"
        log.info("llama-server %s stopped (%s)", owned.pid, outcome)
        return outcome

    def stop(self) -> str | None:
        self._abort.set()  # a start in progress gives up and stops its process
        with self._lock:
            outcome = self._stop_locked()
        self._abort.clear()
        return outcome


def spec_for(
    profile: GenerationProfile,
    executable: str,
    model_path: str,
    alias: str,
    runtime_args: list[str],
) -> LaunchSpec:
    return LaunchSpec(
        executable=executable,
        model_path=model_path,
        alias=alias,
        port=profile.port,
        context=profile.context,
        gpu_layers=profile.gpu_layers,
        threads=profile.threads,
        parallel=profile.parallel,
        flash_attn=profile.flash_attn,
        kv_type=profile.kv_type,
        runtime_args=tuple(runtime_args),
    )
