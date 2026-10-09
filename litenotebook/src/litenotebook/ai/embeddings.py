"""MiniLM: detect, install on request, test in the worker process.

* Detection reads the local Hugging Face cache only (no network, no torch).
* Installation runs ``litenotebook.ai.minilm_fetch`` as a child process with
  network access enabled for that child only, then verifies every required
  file and the weights' SHA-256 against the catalog.
* The test loads MiniLM in the isolated embedding worker (never in the web
  process) and embeds one fixed sentence.
Normal operation keeps loading MiniLM with ``local_files_only``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from litenotebook.ai.catalog import EmbeddingEntry, load_embeddings
from litenotebook.platform_utils.filesystem import COPY_BUFFER_BYTES
from litenotebook.platform_utils.processes import spawn, stop
from litenotebook.platform_utils.system import python_module_command


class EmbeddingSetupError(RuntimeError):
    pass


def entry() -> EmbeddingEntry:
    return load_embeddings()[0]


def cached_files(e: EmbeddingEntry | None = None) -> dict[str, Path | None]:
    e = e or entry()
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return {f.path: None for f in e.files}
    out: dict[str, Path | None] = {}
    for f in e.files:
        found = None
        with contextlib.suppress(Exception):
            hit = try_to_load_from_cache(e.repository, f.path)
            found = Path(hit) if isinstance(hit, str) else None
        out[f.path] = found
    return out


def status() -> dict[str, Any]:
    started = time.perf_counter()
    e = entry()
    files = cached_files(e)
    missing = [
        name for name, path in files.items() if path is None or not path.is_file()
    ]
    return {
        "id": e.id,
        "name": e.display_name,
        "installed": not missing,
        "missing": missing,
        "size": e.size,
        "license": e.license,
        "license_url": e.license_url,
        "model_page": e.model_page,
        "detect_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def verify(e: EmbeddingEntry | None = None) -> None:
    e = e or entry()
    files = cached_files(e)
    for f in e.files:
        path = files.get(f.path)
        if path is None or not path.is_file():
            raise EmbeddingSetupError(f"MiniLM file missing after install: {f.path}")
        if f.sha256:
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                while chunk := fh.read(COPY_BUFFER_BYTES):
                    digest.update(chunk)
            if digest.hexdigest() != f.sha256:
                raise EmbeddingSetupError(
                    f"{f.path} does not match its published checksum; reinstall MiniLM"
                )


def install(
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Explicit download into the HF cache, in a child process."""
    if status()["installed"]:
        verify()
        return {"installed": True, "downloaded": False}
    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    owned = spawn(
        command or python_module_command("litenotebook.ai.minilm_fetch"), env=env
    )
    result: dict[str, Any] = {}
    try:
        assert owned.stdout is not None
        for raw in iter(owned.stdout.readline, b""):
            if cancel is not None and cancel.is_set():
                raise EmbeddingSetupError("Cancelled")
            try:
                message = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                continue
            if progress is not None:
                progress(message)
            if "error" in message or message.get("ok"):
                result = message
                break
    finally:
        stop(
            owned,
            graceful=owned.proc.terminate if cancel and cancel.is_set() else None,
            grace_s=10,
        )
    if "error" in result:
        if result.get("offline"):
            raise EmbeddingSetupError(
                "No internet connection. You can retry later; MiniLM is only needed "
                "for semantic search."
            )
        raise EmbeddingSetupError(str(result["error"]))
    if not result.get("ok"):
        raise EmbeddingSetupError("The MiniLM download stopped unexpectedly")
    verify()
    return {"installed": True, "downloaded": True}


def test(embedder: Any) -> dict[str, Any]:
    """Load MiniLM in the worker process and embed one fixed sentence."""
    started = time.perf_counter()
    embedder.load()
    vectors = embedder.encode(["LiteNotebook test sentence."])
    return {
        "ok": len(vectors) == 1,
        "seconds": round(time.perf_counter() - started, 2),
        "dim": int(vectors.shape[1]) if hasattr(vectors, "shape") else None,
    }
