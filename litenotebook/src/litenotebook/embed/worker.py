"""The isolated MiniLM embedding worker: ``python -m litenotebook.embed.worker``.

LiteNotebook launches this as a child process and talks to it over stdin/stdout
(see ``protocol``). It is the ONLY process that imports torch and
sentence-transformers; when it exits, all of that memory goes back to the OS.

* stdout carries protocol messages only. At start-up the real stdout file
  descriptor is moved aside for the protocol and fd 1 is pointed at stderr,
  so any stray print or C-level write from torch/Hugging Face lands on stderr
  instead of corrupting the IPC stream.
* EOF on stdin (the parent closed the pipe or died) ends the worker. Only the
  parent holds the write end (close_fds), so its death always means EOF.
  (PR_SET_PDEATHSIG is deliberately not used: it fires when the spawning
  *thread* exits, which would kill a healthy worker.)
* The model loads on "hello" from the local Hugging Face cache only (Phase 3
  rules: CPU, local files, no download) and stays loaded until "shutdown".
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Any, BinaryIO, Protocol

import numpy as np
import numpy.typing as npt

from litenotebook.config import EMBED_DIM, EMBED_MODEL
from litenotebook.embed import protocol as p
from litenotebook.embed.embedder import (
    EmbeddingError,
    EmbeddingModelUnavailableError,
    validate_matrix,
)

BACKEND_ENV = "LITENOTEBOOK_EMBED_WORKER_BACKEND"
TEST_MODE_ENV = "LITENOTEBOOK_EMBED_WORKER_TEST_MODE"
TEST_DIR_ENV = "LITENOTEBOOK_EMBED_WORKER_TEST_DIR"
MAX_TOTAL_CHARS = 4 * 1024 * 1024

log = logging.getLogger("litenotebook.embed.worker")


class Backend(Protocol):
    def load(self) -> dict[str, Any]: ...

    def encode(self, texts: list[str]) -> npt.NDArray[np.float32]: ...


class MiniLMBackend:
    """all-MiniLM-L6-v2 on the CPU, exactly as the in-process embedder ran it."""

    def __init__(self) -> None:
        from litenotebook.embed.embedder import MiniLMEmbedder

        self.embedder = MiniLMEmbedder()

    def load(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.embedder.load()
        import torch  # already imported by sentence-transformers

        return {
            "load_s": round(time.perf_counter() - started, 3),
            "device": "cpu",
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }

    def encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return self.embedder.encode(texts)


def _protect_stdout() -> BinaryIO:
    """Return a private handle on the real stdout; make fd 1 point at stderr."""
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(proto_fd, "wb", buffering=0)


def _backend() -> Backend:
    if os.environ.get(BACKEND_ENV) == "test-fake":
        from litenotebook.embed._test_backend import FakeBackend

        return FakeBackend()
    return MiniLMBackend()


def _texts(message: dict[str, Any]) -> list[str]:
    texts = message.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts must be a list of strings")
    if len(texts) > p.MAX_TEXTS_PER_REQUEST:
        raise ValueError(f"at most {p.MAX_TEXTS_PER_REQUEST} texts per request")
    if sum(len(t) for t in texts) > MAX_TOTAL_CHARS:
        raise ValueError("request too large")
    return texts


def serve(stdin: BinaryIO, out: BinaryIO, backend: Backend) -> int:
    loaded = False
    test_mode = os.environ.get(TEST_MODE_ENV, "") if os.environ.get(BACKEND_ENV) else ""

    def send(message: dict[str, Any]) -> None:
        out.write(p.dumps(message))

    for raw in stdin:
        try:
            message = p.loads(raw)
        except p.ProtocolError as e:
            send(p.error(-1, p.ERR_BAD_REQUEST, str(e)))
            continue
        rid, op = message["id"], message.get("op")
        try:
            if op == "hello":
                info = backend.load()
                loaded = True
                send(
                    p.ok(
                        rid,
                        {
                            "protocol": p.PROTOCOL_VERSION,
                            "pid": os.getpid(),
                            "model": EMBED_MODEL,
                            "dim": EMBED_DIM,
                            **info,
                        },
                    )
                )
            elif op == "embed":
                if not loaded:
                    raise ValueError("say hello first")
                texts = _texts(message)
                vectors = (
                    backend.encode(texts)
                    if texts
                    else np.zeros((0, EMBED_DIM), dtype=np.float32)
                )
                validate_matrix(vectors)
                if len(vectors) != len(texts):
                    raise EmbeddingError("vector count does not match text count")
                if test_mode == "garbage":
                    out.write(b"this is not a protocol message\n")
                    continue
                if test_mode == "wrong_id":
                    rid += 1000
                if test_mode == "bad_vectors":
                    vectors = vectors[:, :10]
                send(p.ok(rid, p.encode_vectors(vectors)))
            elif op == "ping":
                send(p.ok(rid, {"loaded": loaded, "pid": os.getpid()}))
            elif op == "shutdown":
                send(p.ok(rid, {}))
                return 0
            else:
                send(p.error(rid, p.ERR_BAD_REQUEST, f"unknown op {op!r}"))
        except EmbeddingModelUnavailableError as e:
            send(p.error(rid, p.ERR_MODEL_UNAVAILABLE, str(e)))
        except EmbeddingError as e:
            send(p.error(rid, p.ERR_EMBEDDING, str(e)))
        except ValueError as e:
            send(p.error(rid, p.ERR_BAD_REQUEST, str(e)))
        except Exception as e:  # report, keep serving
            log.exception("Embedding worker request failed")
            send(p.error(rid, p.ERR_INTERNAL, f"{type(e).__name__}: {e}"[:300]))
    return 0  # EOF: the parent closed the pipe or died


def main() -> int:
    out = _protect_stdout()
    logging.basicConfig(
        stream=sys.stderr, level=logging.WARNING, format="embed-worker: %(message)s"
    )
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C goes to the parent
    try:
        return serve(sys.stdin.buffer, out, _backend())
    except BrokenPipeError:
        return 0  # the parent went away mid-reply


if __name__ == "__main__":
    sys.exit(main())
