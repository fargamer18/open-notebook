"""Platform diagnostics for bug reports (CLI ``diagnostics platform`` and
Settings → Diagnostics).

Read-only and cheap: it never starts the embedding worker, never contacts
the LLM, never imports torch, and never includes secrets (API keys, URL
credentials) or document names. Paths are shown with the home folder as
``~`` so a pasted report does not reveal a user name.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from litenotebook import __version__
from litenotebook.config import EMBED_MODEL, Settings


def redact_path(path: Path | str) -> str:
    """Home folder → ``~`` (both ``/`` and ``\\`` styles)."""
    text = str(path)
    home = str(Path.home())
    if home and home not in ("/", "\\") and text.lower().startswith(home.lower()):
        return "~" + text[len(home) :]
    return text


def redact_url(url: str) -> str:
    """Drop any ``user:password@`` part of an endpoint URL."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "(invalid URL)"
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pdf_engine() -> dict[str, str]:
    from litenotebook.pdf import engine_versions

    return engine_versions()


def _sqlite_features() -> dict[str, Any]:
    conn = sqlite3.connect(":memory:")
    try:
        try:
            conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            fts5 = True
        except sqlite3.OperationalError:
            fts5 = False
        try:
            conn.execute("SELECT json_valid('{}')")
            json1 = True
        except sqlite3.OperationalError:
            json1 = False
    finally:
        conn.close()
    return {"version": sqlite3.sqlite_version, "fts5": fts5, "json1": json1}


def _minilm_cached() -> bool | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        return isinstance(try_to_load_from_cache(EMBED_MODEL, "model.safetensors"), str)
    except Exception:  # cache layout problems must not break diagnostics
        return None


def collect(
    settings: Settings,
    *,
    effective_llm: Mapping[str, Any] | None = None,
    worker_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from litenotebook.packaging import (
        app_dir,
        build_manifest,
        embed_worker_command,
        is_frozen,
        runtime_status,
    )
    from litenotebook.platform_utils.filesystem import InstanceLock, in_use
    from litenotebook.platform_utils.system import (
        find_executable,
        os_name,
        python_module_command,
        system_info,
    )

    info = system_info(settings.data_dir)
    try:
        worker_cmd = [redact_path(p) for p in embed_worker_command(settings)]
    except Exception as e:  # missing runtime / unsupported: say why
        worker_cmd = [f"unavailable: {e}"]
    del python_module_command
    runtime = runtime_status(settings)
    manifest = build_manifest()
    llama = find_executable("llama-server")
    data_in_use = settings.data_dir.exists() and in_use(settings.data_dir)
    holder = InstanceLock(settings.data_dir).holder() if data_in_use else None
    counts: dict[str, int] | None = None
    if settings.db_path.exists():
        # as_uri() percent-encodes '#'/'?' and gives file:///C:/… on Windows
        uri = settings.db_path.absolute().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.execute("PRAGMA query_only = ON")
            counts = {
                "schema_version": conn.execute("PRAGMA user_version").fetchone()[0],
                "notebooks": conn.execute("SELECT count(*) FROM notebooks").fetchone()[
                    0
                ],
                "sources": conn.execute("SELECT count(*) FROM sources").fetchone()[0],
            }
        except sqlite3.Error:
            counts = None
        finally:
            conn.close()
    llm = dict(effective_llm or {})
    return {
        "litenotebook": __version__,
        "application": {
            "version": __version__,
            "mode": "packaged" if is_frozen() else "source",
            "executable": redact_path(sys.executable),
            "app_folder": redact_path(folder) if (folder := app_dir()) else None,
            "build": manifest.get("built_at"),
            "build_platform": manifest.get("platform"),
        },
        "platform": {
            "os": f"{info.os} {info.os_release}",
            "family": os_name(),
            "architecture": info.architecture,
            "python": f"{info.python_implementation} {info.python_version}",
            "cpu": info.cpu_name,
            "logical_cpus": info.logical_cpus,
            "physical_cpus": info.physical_cpus,
            "ram_total_gib": round(info.total_ram_bytes / 2**30, 1)
            if info.total_ram_bytes
            else None,
            "ram_available_gib": round(info.available_ram_bytes / 2**30, 1)
            if info.available_ram_bytes
            else None,
            "disk_free_gib": round(info.disk_free_bytes / 2**30, 1)
            if info.disk_free_bytes is not None
            else None,
        },
        "paths": {
            "data": redact_path(settings.data_dir),
            "data_origin": settings.data_dir_origin,
            "config": redact_path(settings.config_path),
            "cache": redact_path(settings.cache_path),
            "logs": redact_path(settings.log_path),
            "data_in_use": data_in_use,
            "in_use_by": {"pid": holder.get("pid"), "purpose": holder.get("purpose")}
            if holder
            else None,
        },
        "database": {"sqlite": _sqlite_features(), **(counts or {"exists": False})},
        "libraries": {
            "pdf_engine": _pdf_engine(),
            "faiss": _version("faiss-cpu") or _version("faiss"),
            "faiss_importable": importlib.util.find_spec("faiss") is not None,
            "numpy": _version("numpy"),
            "fastapi": _version("fastapi"),
            "uvicorn": _version("uvicorn"),
            "torch_in_this_process": "torch" in sys.modules,  # must stay False
        },
        "embedding_worker": {
            "command": worker_cmd,
            "runtime": "source checkout"
            if not runtime["packaged"]
            else (runtime["where"] or "not installed"),
            "model": EMBED_MODEL,
            "model_cached_locally": _minilm_cached(),
            "status": dict(worker_status)
            if worker_status
            else {"status": "not running"},
        },
        "llm": {
            "endpoint": redact_url(str(llm.get("base_url", settings.llm_base_url))),
            "model": llm.get("model", settings.llm_model),
            "api_key_configured": bool(llm.get("api_key_configured", False)),
            "llama_server_on_path": redact_path(llama) if llama else None,
        },
    }


def format_text(report: Mapping[str, Any]) -> str:
    lines = [f"LiteNotebook {report['litenotebook']} diagnostics"]
    for section, values in report.items():
        if not isinstance(values, Mapping):
            continue
        lines.append(f"\n[{section}]")
        for key, value in values.items():
            if isinstance(value, Mapping):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            elif isinstance(value, list):
                value = " ".join(str(v) for v in value)
            lines.append(f"{key}: {value if value is not None else 'unknown'}")
    return "\n".join(lines) + "\n"
