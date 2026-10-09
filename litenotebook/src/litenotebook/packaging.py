"""Packaged (frozen) builds versus a source checkout.

A desktop build is a PyInstaller one-folder app::

    LiteNotebook/                      (Windows: LiteNotebook.exe)
      litenotebook                     the web app and launcher (no torch)
      _internal/                       Python runtime, libraries, templates
      litenotebook-build.json          version, platform, expected companions
      embedding-runtime/               optional: the MiniLM worker, beside the app
        litenotebook-embed-worker      (Windows: .exe) + its own _internal/

The embedding worker (torch, sentence-transformers) is a separate optional
component so the core download stays small. It is found, in order, at
``LITENOTEBOOK_EMBED_WORKER``, beside the app (``embedding-runtime/``), or
installed into the data folder from the verified runtime archive
(``<data>/runtime/embedding/<version>/``). In a source checkout the worker
is ``python -m litenotebook.embed.worker`` as before.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from litenotebook import __version__
from litenotebook.config import Settings
from litenotebook.embed.embedder import EmbeddingModelUnavailableError
from litenotebook.platform_utils.filesystem import is_within, remove_tree, rename_dir
from litenotebook.platform_utils.system import (
    executable_name,
    machine_arch,
    os_name,
    python_module_command,
)

APP_NAME = "LiteNotebook"
WORKER_BASE = "litenotebook-embed-worker"
RUNTIME_DIRNAME = "embedding-runtime"
BUILD_MANIFEST = "litenotebook-build.json"
RUNTIME_MARKER = "litenotebook-embedding-runtime.json"
WORKER_ENV = "LITENOTEBOOK_EMBED_WORKER"


class EmbeddingRuntimeMissingError(EmbeddingModelUnavailableError):
    """Packaged build without the optional embedding runtime."""

    def __init__(self) -> None:
        Exception.__init__(
            self,
            "The semantic search runtime is not installed. Install it from "
            "Settings → Local AI (Semantic search); LiteNotebook needs it to "
            "index PDFs.",
        )


class RuntimeInstallError(ValueError):
    """Message safe to show the user."""


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path | None:
    """The folder holding the packaged executable (None in a source checkout)."""
    return Path(sys.executable).resolve().parent if is_frozen() else None


def build_manifest() -> dict[str, Any]:
    folder = app_dir()
    if folder is None:
        return {}
    try:
        data = json.loads((folder / BUILD_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def platform_tag() -> str:
    arch = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine_arch().lower(), machine_arch().lower())
    return f"{os_name()}-{arch}"


def installed_runtime_dir(settings: Settings) -> Path:
    return settings.data_dir / "runtime" / "embedding" / __version__


def _worker_in(folder: Path) -> Path | None:
    exe = folder / executable_name(WORKER_BASE)
    return exe if exe.is_file() else None


def find_embed_worker(
    settings: Settings, env: Mapping[str, str] | None = None
) -> tuple[Path, str] | None:
    """(worker executable, where it was found) for a packaged build."""
    env = os.environ if env is None else env
    explicit = env.get(WORKER_ENV, "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        return (path, "LITENOTEBOOK_EMBED_WORKER") if path.is_file() else None
    folder = app_dir()
    if folder is not None and (exe := _worker_in(folder / RUNTIME_DIRNAME)):
        return exe, "beside the app"
    installed = installed_runtime_dir(settings)
    marker = _read_marker(installed)
    if marker.get("version") == __version__ and (exe := _worker_in(installed)):
        return exe, "installed in the data folder"
    return None


def _read_marker(folder: Path) -> dict[str, Any]:
    try:
        data = json.loads((folder / RUNTIME_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def embed_worker_command(settings: Settings) -> list[str]:
    """How to start the embedding worker in this build."""
    if not is_frozen():
        return python_module_command("litenotebook.embed.worker")
    found = find_embed_worker(settings)
    if found is None:
        raise EmbeddingRuntimeMissingError()
    return [str(found[0])]


def runtime_status(settings: Settings) -> dict[str, Any]:
    """For Settings and Diagnostics. Reads files only."""
    expected = build_manifest().get("embedding_runtime") or {}
    if not is_frozen():
        return {
            "packaged": False,
            "installed": True,
            "where": "source checkout",
            "expected": None,
        }
    found = find_embed_worker(settings)
    return {
        "packaged": True,
        "installed": found is not None,
        "where": found[1] if found else None,
        "expected": expected or None,
        "version": __version__,
    }


def install_runtime_archive(
    settings: Settings,
    archive: Path,
    *,
    progress: Callable[[str], None] | None = None,
    expected: Mapping[str, Any] | None = None,
) -> Path:
    """Install the embedding runtime from its release archive (a local file the
    user chose). The archive must match the name, size and SHA-256 recorded
    in this build's manifest, so only the companion of this exact version is
    accepted; it is extracted with the hostile-archive rules."""
    from litenotebook.ai.extract import extract

    say = progress or (lambda _m: None)
    expected = dict(
        expected
        if expected is not None
        else build_manifest().get("embedding_runtime") or {}
    )
    if not expected:
        raise RuntimeInstallError(
            "This build does not list an embedding runtime to install"
        )
    archive = Path(archive).expanduser()
    if not archive.is_file():
        raise RuntimeInstallError("That file does not exist")
    if archive.name != expected.get("file"):
        raise RuntimeInstallError(
            f"Choose {expected.get('file')} (the runtime for LiteNotebook "
            f"{__version__} on this system)"
        )
    if archive.stat().st_size != expected.get("size"):
        raise RuntimeInstallError("The file has the wrong size (incomplete download?)")
    say("Checking the file…")
    digest = hashlib.sha256()
    with open(archive, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != expected.get("sha256"):
        raise RuntimeInstallError(
            "The file does not match the published checksum; download it again"
        )
    target = installed_runtime_dir(settings)
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging-{uuid.uuid4().hex[:12]}"
    fmt = "zip" if archive.name.endswith(".zip") else "tar.gz"
    try:
        say("Extracting…")
        extract(archive, fmt, staging)
        inner = [p for p in staging.iterdir() if p.is_dir()]
        base = inner[0] if len(inner) == 1 and not _worker_in(staging) else staging
        if not _worker_in(base) or not is_within(base, staging):
            raise RuntimeInstallError("The archive does not contain the worker program")
        if os_name() != "windows":
            exe = _worker_in(base)
            assert exe is not None
            exe.chmod(exe.stat().st_mode | 0o111)
        (base / RUNTIME_MARKER).write_text(
            json.dumps(
                {
                    "version": __version__,
                    "sha256": digest.hexdigest(),
                    "installed_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        old = None
        if target.exists():
            old = root / f".old-{uuid.uuid4().hex[:12]}"
            rename_dir(target, old)
        rename_dir(base, target)
        if old is not None:
            with contextlib.suppress(OSError):
                remove_tree(old)
    finally:
        with contextlib.suppress(OSError):
            if staging.exists():
                remove_tree(staging)
    say("Installed")
    return target
