"""Parent side of the isolated embedding worker.

``WorkerEmbedder`` is a ``TextEmbedder`` whose MiniLM runs in ONE child process
(``python -m litenotebook.embed.worker``), so torch / sentence-transformers are
never imported into LiteNotebook itself and their ~500 MB is returned to the OS
whenever the worker exits.

Lifecycle::

    stopped ──first embed/load──▶ starting ──hello ok──▶ ready ⇄ busy
       ▲                              │ (load failed)        │
       │                              ▼                      │ idle timeout
       └──────────── stopping ◀── failed / crash ◀───────────┘

* Lazy: nothing starts until an embedding is needed; ``status()`` never starts it.
* One worker: process start/stop and requests all run under one lease, so two
  first requests can never spawn two workers.
* Scheduling: each batch takes the lease separately, so a query can run between
  an ingestion job's batches. Waiting requests are bounded; single-text
  (interactive) requests go before multi-text (ingestion) batches, FIFO within
  a priority. A batch already running is never pre-empted.
* Failures (crash, broken pipe, timeout, malformed reply): the child is killed
  and reaped, state resets, and an embedding is retried ONCE on a fresh worker.
  Structured worker errors (missing model, oversized text) are not retried.
* Idle: after ``idle_seconds`` without work the parent asks the worker to exit
  (terminate → kill only as fallbacks) and reaps it.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import heapq
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import weakref
from typing import IO, Any, ClassVar

import numpy as np
import numpy.typing as npt

from litenotebook.config import EMBED_DIM, EMBED_MODEL
from litenotebook.embed import protocol as p
from litenotebook.embed.embedder import (
    EmbeddingError,
    EmbeddingModelUnavailableError,
    validate_matrix,
)

log = logging.getLogger(__name__)

DEFAULT_IDLE_SECONDS = 120.0
DEFAULT_START_TIMEOUT = 180.0  # observed cold load ~15 s on the target laptop
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_BATCH = 64
DEFAULT_MAX_WAITING = 8
STOP_GRACE_S = 5.0
INTERACTIVE, BACKGROUND = 0, 1


class EmbeddingWorkerError(EmbeddingError):
    """The worker could not produce embeddings (message is safe for the UI)."""


class EmbeddingBusyError(EmbeddingWorkerError):
    pass


class _Broken(Exception):
    """The worker process is unusable; kill it (and maybe retry once)."""


class WorkerEmbedder:
    model_name = EMBED_MODEL
    dim = EMBED_DIM
    thread_safe = True  # does its own scheduling; callers need no lock

    _instances: ClassVar[weakref.WeakSet[WorkerEmbedder]] = weakref.WeakSet()

    def __init__(
        self,
        *,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        batch_size: int = DEFAULT_BATCH,
        max_waiting: int = DEFAULT_MAX_WAITING,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.idle_seconds = float(idle_seconds)
        self.start_timeout = float(start_timeout)
        self.request_timeout = float(request_timeout)
        self.batch_size = max(1, int(batch_size))
        self.max_waiting = max(1, int(max_waiting))
        self.command = command or [sys.executable, "-m", "litenotebook.embed.worker"]
        self.extra_env = dict(env or {})

        self._cv = threading.Condition()
        self._leased = False
        self._waiting: list[tuple[int, int]] = []
        self._tickets = 0
        self._closed = False

        self._proc: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[bytes | None] | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=20)
        self._state = "stopped"
        self._next_id = 0
        self._last_used = time.monotonic()
        self._last_error: str | None = None
        self.info: dict[str, Any] = {}
        self.spawns = 0
        self._watcher: threading.Thread | None = None

        WorkerEmbedder._instances.add(self)
        atexit.register(_close_ref, weakref.ref(self))

    # --- public -----------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._alive() and self._state in ("ready", "busy")

    def load(self) -> None:
        """Start the worker (and load MiniLM) now, if it is not running."""
        self._lease(INTERACTIVE)
        try:
            if not self._alive():
                self._start()
        except _Broken as e:
            self._kill()
            raise EmbeddingWorkerError(str(e)) from e
        finally:
            self._unlease()

    def encode(
        self, texts: list[str], *, interactive: bool | None = None
    ) -> npt.NDArray[np.float32]:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        if interactive is None:
            interactive = len(texts) == 1  # a query, not an ingestion batch
        priority = INTERACTIVE if interactive else BACKGROUND
        parts = [
            self._embed_batch(texts[i : i + self.batch_size], priority)
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def status(self) -> dict[str, Any]:
        """Never starts the worker."""
        with self._cv:
            alive = self._alive()
            state = self._state if alive or self._state != "ready" else "stopped"
            out: dict[str, Any] = {"status": state}
            if alive and self._proc is not None:
                out["pid"] = self._proc.pid
                if state == "ready":
                    out["idle_seconds"] = round(time.monotonic() - self._last_used, 1)
            out["idle_timeout_seconds"] = self.idle_seconds
            out["waiting"] = len(self._waiting)
            if self._last_error and state == "failed":
                out["last_error"] = self._last_error
            return out

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    def close(self) -> None:
        """Stop the worker and reap it (server shutdown / CLI exit)."""
        with self._cv:
            if self._closed:
                return
            self._closed = True
            deadline = time.monotonic() + STOP_GRACE_S
            while self._leased and time.monotonic() < deadline:
                self._cv.wait(0.1)
            self._leased = True
            self._cv.notify_all()
        try:
            self._stop()
        finally:
            with self._cv:
                self._leased = False
                self._cv.notify_all()

    @classmethod
    def close_all(cls) -> None:
        for instance in list(cls._instances):
            instance.close()

    # --- scheduling -------------------------------------------------------------

    def _lease(self, priority: int) -> None:
        with self._cv:
            if self._closed:
                raise EmbeddingWorkerError("The embedding worker has been shut down")
            if len(self._waiting) >= self.max_waiting:
                raise EmbeddingBusyError(
                    "The embedding model is busy with too many requests; retry shortly"
                )
            self._tickets += 1
            entry = (priority, self._tickets)
            heapq.heappush(self._waiting, entry)
            try:
                while self._leased or self._waiting[0] != entry:
                    self._cv.wait()
                    if self._closed:
                        raise EmbeddingWorkerError(
                            "The embedding worker has been shut down"
                        )
            except BaseException:
                self._waiting.remove(entry)
                heapq.heapify(self._waiting)
                self._cv.notify_all()
                raise
            heapq.heappop(self._waiting)
            self._leased = True

    def _unlease(self) -> None:
        with self._cv:
            self._leased = False
            self._last_used = time.monotonic()
            self._cv.notify_all()

    # --- process ------------------------------------------------------------------

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                # child-only safety nets; the parent's environment is untouched
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.update(self.extra_env)
        return env

    def _start(self) -> None:
        self._state = "starting"
        self.spawns += 1
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._child_env(),
                close_fds=True,
            )
        except OSError as e:
            self._state = "failed"
            self._last_error = "The embedding worker could not be started"
            raise _Broken(self._last_error) from e
        self._proc = proc
        self._lines = queue.Queue()
        assert proc.stdout is not None and proc.stderr is not None
        threading.Thread(
            target=_read_lines, args=(proc.stdout, self._lines), daemon=True
        ).start()
        threading.Thread(
            target=_drain_stderr, args=(proc.stderr, self._stderr_tail), daemon=True
        ).start()
        try:
            info = self._call("hello", self.start_timeout)
        except EmbeddingError as e:  # e.g. the model is not available locally
            self._kill()
            self._state = "failed"
            self._last_error = str(e)
            raise
        if info.get("protocol") != p.PROTOCOL_VERSION or info.get("dim") != EMBED_DIM:
            raise _Broken("The embedding worker speaks a different protocol")
        if info.get("model") != EMBED_MODEL:
            raise _Broken("The embedding worker loaded a different model")
        self.info = {**info, "spawn_s": round(time.perf_counter() - started, 3)}
        self._state = "ready"
        self._last_error = None
        self._ensure_watcher()
        log.info("Embedding worker %s ready in %.1fs", proc.pid, self.info["spawn_s"])

    def _call(self, op: str, timeout: float, **fields: Any) -> dict[str, Any]:
        proc, lines = self._proc, self._lines
        if proc is None or lines is None or proc.stdin is None:
            raise _Broken("The embedding worker is not running")
        self._next_id += 1
        rid = self._next_id
        try:
            proc.stdin.write(p.dumps(p.request(rid, op, **fields)))
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise _Broken(self._exit_message()) from e
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty as e:
            raise _Broken(
                f"The embedding worker did not answer within {timeout:.0f} s"
            ) from e
        if line is None:
            raise _Broken(self._exit_message())
        try:
            message = p.loads(line)
        except p.ProtocolError as e:
            raise _Broken("The embedding worker sent an invalid reply") from e
        if message["id"] != rid:
            raise _Broken("The embedding worker answered the wrong request")
        if message.get("ok") is True:
            result = message.get("result")
            if not isinstance(result, dict):
                raise _Broken("The embedding worker sent an invalid reply")
            return result
        raw_error = message.get("error")
        err: dict[str, Any] = raw_error if isinstance(raw_error, dict) else {}
        kind, text = err.get("type"), str(err.get("message", "unknown error"))
        if kind == p.ERR_MODEL_UNAVAILABLE:
            raise EmbeddingModelUnavailableError()
        if kind == p.ERR_EMBEDDING:
            raise EmbeddingError(text)
        raise EmbeddingWorkerError(f"The embedding worker reported: {text[:200]}")

    def _exit_message(self) -> str:
        proc = self._proc
        code = None
        if proc is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                code = proc.wait(timeout=1.0)
        detail = f" (exit code {code})" if code is not None else ""
        return f"The embedding worker stopped unexpectedly{detail}"

    def _embed_batch(self, texts: list[str], priority: int) -> npt.NDArray[np.float32]:
        self._lease(priority)
        try:
            for attempt in (1, 2):
                try:
                    if not self._alive():
                        self._start()
                    self._state = "busy"
                    result = self._call("embed", self.request_timeout, texts=texts)
                    vectors = p.decode_vectors(result, len(texts))
                    validate_matrix(vectors)
                    self._state = "ready"
                    return vectors
                except (_Broken, p.ProtocolError) as e:
                    log.warning("Embedding worker failed (attempt %d): %s", attempt, e)
                    self._last_error = str(e)
                    self._kill()
                    self._state = "failed"
                    if attempt == 2:
                        raise EmbeddingWorkerError(f"{e}. Retry.") from e
                except EmbeddingError:
                    if self._alive():
                        self._state = "ready"
                    raise
            raise AssertionError("unreachable")
        finally:
            self._unlease()

    def _stop(self) -> None:
        proc = self._proc
        if proc is None:
            self._state = "stopped"
            return
        self._state = "stopping"
        if proc.poll() is None:
            with contextlib.suppress(_Broken, EmbeddingError):
                self._call("shutdown", STOP_GRACE_S)
        self._kill()
        self._state = "stopped"

    def _kill(self) -> None:
        """Make sure the child is gone and reaped (no zombies, no orphans)."""
        proc, self._proc = self._proc, None
        self._lines = None
        if proc is None:
            return
        for pipe in (proc.stdin,):
            with contextlib.suppress(OSError, ValueError):
                if pipe is not None:
                    pipe.close()
        try:
            proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for pipe in (proc.stdout, proc.stderr):
            with contextlib.suppress(OSError, ValueError):
                if pipe is not None:
                    pipe.close()
        if proc.returncode not in (0, None):
            log.info("Embedding worker %s exited with %s", proc.pid, proc.returncode)

    # --- idle shutdown ----------------------------------------------------------

    def _ensure_watcher(self) -> None:
        if self._watcher is not None and self._watcher.is_alive():
            return
        self._watcher = threading.Thread(
            target=_watch_idle,
            args=(weakref.ref(self),),
            daemon=True,
            name="litenotebook-embed-idle",
        )
        self._watcher.start()

    def _maybe_idle_stop(self) -> bool:
        """Stop the worker if it has been idle long enough. Returns False when
        there is nothing left to watch."""
        with self._cv:
            if self._closed or not self._alive():
                return False
            idle = time.monotonic() - self._last_used
            if self._leased or self._waiting or idle < self.idle_seconds:
                return True
            self._leased = True  # nobody can start a request while we stop
        try:
            log.info("Embedding worker idle for %.0fs; stopping it", idle)
            self._stop()
        finally:
            self._unlease()
        return False


def _watch_idle(ref: weakref.ref[WorkerEmbedder]) -> None:
    while True:
        embedder = ref()
        if embedder is None:
            return
        interval = max(0.05, min(1.0, embedder.idle_seconds / 4))
        del embedder
        time.sleep(interval)
        embedder = ref()
        if embedder is None or not embedder._maybe_idle_stop():
            return
        del embedder


def _read_lines(stream: IO[bytes], lines: queue.Queue[bytes | None]) -> None:
    try:
        while True:
            line = stream.readline(p.MAX_LINE_BYTES + 1)
            if not line:
                break
            lines.put(line)
    except (OSError, ValueError):
        pass
    finally:
        lines.put(None)


def _drain_stderr(stream: IO[bytes], tail: collections.deque[str]) -> None:
    with contextlib.suppress(OSError, ValueError):
        for raw in stream:
            text = raw.decode("utf-8", "replace").rstrip()
            if text:
                tail.append(text)
                log.debug("embed-worker: %s", text)


def _close_ref(ref: weakref.ref[WorkerEmbedder]) -> None:
    embedder = ref()
    if embedder is not None:
        with contextlib.suppress(Exception):
            embedder.close()
