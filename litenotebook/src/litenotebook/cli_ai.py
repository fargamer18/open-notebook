"""``litenotebook ai …``: the same Local AI actions as the setup wizard and
Settings → Local AI (``LocalAI`` in ``ai.service``), for scripts and
terminals.

Downloads happen only through ``install`` commands, after the size, source
and license are printed; without ``--yes`` they ask first (and refuse when
there is no terminal to ask on). A llama-server started by a test is
stopped before the command exits.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from typing import Any

from litenotebook.ai import catalog as cat
from litenotebook.ai import embeddings as emb
from litenotebook.ai.download import DownloadError, Progress, human
from litenotebook.ai.models import model_status
from litenotebook.ai.runtime_manager import RuntimeManager
from litenotebook.ai.runtimes import discover_existing, list_installed, probe
from litenotebook.ai.service import LocalAI, SetupError
from litenotebook.ai.store import AIStore
from litenotebook.config import Settings
from litenotebook.services.settings import SettingsService

Args = argparse.Namespace


def _service(settings: Settings) -> LocalAI:
    return LocalAI(settings, RuntimeManager(settings))


def _confirm(args: Args, what: str) -> bool:
    if getattr(args, "yes", False):
        return True
    if not sys.stdin.isatty():
        print("error: add --yes to confirm the download", file=sys.stderr)
        return False
    answer = input(f"{what}\nDownload now? [y/N] ").strip().lower()
    return answer in ("y", "yes")


class _Bar:
    """Single-line byte progress on stderr; real numbers only."""

    def __init__(self) -> None:
        self.last = 0.0

    def __call__(self, stage: str, value: Progress | None = None) -> None:
        now = time.monotonic()
        if value is None:
            print(f"\r{stage}…".ljust(72), file=sys.stderr)
            return
        if now - self.last < 0.5 and value.done != value.total:
            return
        self.last = now
        total = f" of {human(value.total)}" if value.total else ""
        rate = f" · {human(int(value.rate_bps))}/s" if value.rate_bps else ""
        line = f"\r{stage}: {human(value.done)}{total}{rate}"
        print(line.ljust(72), end="", file=sys.stderr, flush=True)


def _dump(value: Any) -> None:
    print(json.dumps(value, indent=1, default=str))


# --- read-only ------------------------------------------------------------------------


def cmd_status(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    service = _service(settings)
    o = service.overview(conn)
    profile, active = o["profile"], o["active_model"]
    if args.json:
        _dump(
            {
                "setup": o["setup"].__dict__,
                "profile": {**profile.__dict__},
                "active_model": active.__dict__ if active else None,
                "embeddings": o["embeddings"],
                "models_dir": str(o["models_dir"]),
            }
        )
        return 0
    print(f"Setup:       {'complete' if o['setup'].completed else 'not finished'}")
    print(f"Generation:  {profile.type}")
    if profile.runtime_version:
        print(f"Runtime:     {profile.runtime_version} ({profile.backend})")
    if profile.launches_runtime:
        print(f"Model:       {active.display_name if active else 'none selected'}")
        layers = "all" if profile.gpu_layers >= 999 else str(profile.gpu_layers)
        print(
            f"Launch:      context {profile.context}, GPU layers {layers}, "
            f"threads {profile.threads}, KV {profile.kv_type}, port {profile.port}"
        )
    if profile.last_test:
        t = profile.last_test
        print(
            f"Last test:   {'passed' if t.get('ok') else 'failed'}"
            + (f", {t['tokens_per_s']} tok/s" if t.get("tokens_per_s") else "")
            + (f" — {t['error']}" if t.get("error") else "")
        )
    e = o["embeddings"]
    print(f"Embeddings:  {'installed' if e['installed'] else 'not installed'}")
    print(f"Models dir:  {o['models_dir']}")
    return 0


def cmd_hardware(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    hw = _service(settings).hardware(conn, rescan=True)
    if args.json:
        _dump(hw.as_dict())
        return 0
    print(f"System:     {hw.platform} {hw.os_release} {hw.architecture}")
    print(
        f"CPU:        {hw.cpu or 'unknown'} "
        f"({hw.physical_cores or '?'} cores / {hw.logical_cores or '?'} threads)"
    )
    print(
        f"RAM:        {human(hw.total_ram_bytes or 0)} total, "
        f"{human(hw.available_ram_bytes or 0)} available"
    )
    print(f"Free disk:  {human(hw.free_disk_bytes or 0)}")
    for g in hw.gpus:
        vram = (
            f"{human(g.vram_bytes)} VRAM ({g.confidence})"
            if g.vram_bytes
            else "VRAM unknown"
        )
        free = f", {human(g.vram_free_bytes)} free" if g.vram_free_bytes else ""
        kind = " integrated" if g.integrated else ""
        print(f"GPU:        {g.name}{kind} — {vram}{free} via {g.source}")
    if not hw.gpus:
        print("GPU:        none detected")
    print(f"Scan:       {hw.scan_ms:.0f} ms")
    return 0


def cmd_runtime_list(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    service = _service(settings)
    print("Pinned builds for this computer:")
    for o in service.runtime_options(conn):
        r = o["entry"]
        size = human(sum(a.size for a in r.assets))
        flags = [
            "installed" if o["installed"] else "",
            "" if o["suitable"] else o["why"],
            "" if r.tested else "untested",
        ]
        extra = ", ".join(f for f in flags if f)
        print(f"  {r.id}  {size}" + (f"  ({extra})" if extra else ""))
    installed = list_installed(settings)
    if installed:
        print("Installed by LiteNotebook:")
        for i in installed:
            print(f"  {i.id}  build {i.version}  {i.executable}")
    found = [probe(p.executable) for p in discover_existing()]  # with devices
    if found:
        print("Found on this computer:")
        for p in found:
            state = f"build {p.build} ({p.backend})" if p.ok else f"unusable: {p.error}"
            print(f"  {p.executable}  {state}")
    return 0


def cmd_runtime_test(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    profile = AIStore(conn).profile()
    path = args.path
    if path is None:
        if not profile.launches_runtime:
            print("No llama-server is configured.", file=sys.stderr)
            return 1
        try:
            path = _service(settings)._executable(profile)
        except SetupError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    result = probe(path)
    if not result.ok:
        print(f"error: {result.error}", file=sys.stderr)
        return 1
    print(f"{result.executable}: version {result.version} build {result.build}")
    devices = ", ".join(result.devices or []) or "CPU"
    print(f"Backend: {result.backend}; devices: {devices}")
    return 0


def cmd_model_list(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    service = _service(settings)
    store = AIStore(conn)
    selected = store.profile().model_id
    print("Installed:")
    for m in store.models():
        mark = "*" if m.id == selected else " "
        where = "owned" if m.owned else "in place"
        print(f" {mark} {m.id}  {human(m.size)}  {where}  {model_status(m)}  {m.path}")
    if not store.models():
        print("   (none)")
    print("Catalog:")
    fits = {}
    if store.hardware():
        fits = {f.model_id: f for f in service.recommendations(conn)}
    for entry in cat.load_models():
        fit = fits.get(entry.id)
        rating = ""
        if fit:
            rating = f"  [{'best fit, ' if fit.recommended else ''}{fit.rating}]"
        print(f"   {entry.id}  {human(entry.size)}  {entry.license}{rating}")
        if fit and args.verbose_reasons:
            for reason in fit.reasons:
                print(f"       {reason}")
    if not fits:
        print("   (run `litenotebook ai hardware` for recommendations)")
    return 0


def cmd_embeddings_status(
    args: Args, conn: sqlite3.Connection, settings: Settings
) -> int:
    status = emb.status()
    if args.json:
        _dump(status)
        return 0
    if status["installed"]:
        print(f"{status['name']}: installed (detected in {status['detect_ms']} ms)")
    else:
        print(
            f"{status['name']}: not installed; missing {', '.join(status['missing'])}"
        )
    return 0


# --- actions --------------------------------------------------------------------------


def cmd_runtime_install(
    args: Args, conn: sqlite3.Connection, settings: Settings
) -> int:
    entry = cat.runtime(args.runtime_id)
    size = sum(a.size for a in entry.assets)
    what = (
        f"llama.cpp {entry.version} ({entry.backend}) — {human(size)} from "
        f"{entry.release_url}\nLicense: {entry.license}"
    )
    if not _confirm(args, what):
        return 1
    try:
        profile = _service(settings).install_runtime(conn, entry.id, progress=_Bar())
    except DownloadError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    print(f"\nInstalled; Local AI now uses {profile.executable}")
    return 0


def cmd_runtime_use(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    profile = _service(settings).use_existing_runtime(conn, args.path)
    print(f"Using {profile.executable} ({profile.runtime_version}, {profile.backend})")
    return 0


def cmd_model_install(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    entry = cat.model(args.model_id)
    what = (
        f"{entry.display_name} — {human(entry.size)} from {entry.url}\n"
        f"License: {entry.license} ({entry.license_url})"
    )
    if not _confirm(args, what):
        return 1
    bar = _Bar()
    try:
        record = _service(settings).install_model(
            conn, entry.id, progress=lambda p: bar("Downloading", p)
        )
    except DownloadError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    print(f"\nInstalled and selected {record.display_name}: {record.path}")
    return 0


def cmd_model_import(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    record = _service(settings).import_model(conn, str(args.path), copy=args.copy)
    how = "copied to" if record.owned else "used in place at"
    print(f"Added and selected {record.display_name} ({how} {record.path})")
    if record.verified:
        print("Checksum matches the catalog entry.")
    return 0


def cmd_model_select(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    _service(settings).select_model(conn, args.model_id)
    print(f"Selected {args.model_id}")
    return 0


def cmd_model_remove(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    record = AIStore(conn).model(args.model_id)
    if record is None:
        print("error: that model is not installed", file=sys.stderr)
        return 1
    if record.owned and not args.yes:
        print(f"error: add --yes to delete {record.path}", file=sys.stderr)
        return 1
    deleted = _service(settings).remove_model(conn, args.model_id)
    print("Deleted the model file." if deleted else "Removed from the list only.")
    return 0


def cmd_embeddings_install(
    args: Args, conn: sqlite3.Connection, settings: Settings
) -> int:
    status = emb.status()
    if status["installed"]:
        print(f"{status['name']} is already installed; nothing to download.")
        AIStore(conn).save_embeddings({"state": "installed"})
        return 0
    entry = emb.entry()
    what = (
        f"{entry.display_name} — {human(entry.size)} from huggingface.co/"
        f"{entry.repository}\nLicense: {entry.license}"
    )
    if not _confirm(args, what):
        return 1

    def progress(message: dict[str, Any]) -> None:
        if message.get("file"):
            print(f"  {message['file']} ({message['index']}/{message['total']})")

    _service(settings).install_embeddings(conn, progress=progress)
    print("Installed and verified.")
    return 0


def _run_test(settings: Settings, conn: sqlite3.Connection, *, benchmark: bool) -> int:
    service = _service(settings)
    legacy = SettingsService(conn).effective()
    try:
        outcome = service.test(
            conn,
            legacy,
            progress=lambda m: print(m, file=sys.stderr),
            benchmark=benchmark,
        )
    finally:
        service.stop()  # never leave a llama-server behind this command
    print(outcome.message)
    for label, value in (
        ("Reply", outcome.reply),
        ("Load", f"{outcome.load_s} s" if outcome.load_s is not None else None),
        ("First token", f"{outcome.ttft_s} s" if outcome.ttft_s is not None else None),
        ("Speed", f"{outcome.tokens_per_s} tokens/s" if outcome.tokens_per_s else None),
        ("RAM", human(outcome.rss_bytes) if outcome.rss_bytes else None),
    ):
        if value:
            print(f"  {label}: {value}")
    for change in outcome.changes or []:
        print(f"  Adjusted: {change}")
    return 0 if outcome.ok else 1


def cmd_test(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    return _run_test(settings, conn, benchmark=False)


def cmd_benchmark(args: Args, conn: sqlite3.Connection, settings: Settings) -> int:
    return _run_test(settings, conn, benchmark=True)


# --- parser ---------------------------------------------------------------------------


def register(sub: Any) -> None:
    ai = sub.add_parser("ai", help="local AI: hardware, llama.cpp, models, MiniLM")
    ai_sub = ai.add_subparsers(dest="ai_command", required=True)

    p = ai_sub.add_parser("status", help="what is configured (no network, no scan)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)
    p = ai_sub.add_parser("hardware", help="scan CPU, RAM, disk and GPUs")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_hardware, mutates=True)

    runtime = ai_sub.add_parser("runtime", help="llama.cpp runtimes")
    rt = runtime.add_subparsers(dest="runtime_command", required=True)
    rt.add_parser("list", help="pinned, installed and found").set_defaults(
        func=cmd_runtime_list
    )
    p = rt.add_parser("install", help="download and install a pinned build")
    p.add_argument("runtime_id")
    p.add_argument("--yes", action="store_true", help="do not ask before downloading")
    p.set_defaults(func=cmd_runtime_install, mutates=True)
    p = rt.add_parser("use", help="use an existing llama-server")
    p.add_argument("path")
    p.set_defaults(func=cmd_runtime_use, mutates=True)
    p = rt.add_parser("test", help="run llama-server --version (no model)")
    p.add_argument("path", nargs="?")
    p.set_defaults(func=cmd_runtime_test)

    model = ai_sub.add_parser("model", help="generation models (GGUF)")
    md = model.add_subparsers(dest="model_command", required=True)
    p = md.add_parser("list", help="installed models and the catalog")
    p.add_argument("--reasons", dest="verbose_reasons", action="store_true")
    p.set_defaults(func=cmd_model_list)
    p = md.add_parser("install", help="download a catalog model")
    p.add_argument("model_id")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_model_install, mutates=True)
    p = md.add_parser("import", help="add a .gguf file you already have")
    p.add_argument("path")
    p.add_argument("--copy", action="store_true", help="copy into the models folder")
    p.set_defaults(func=cmd_model_import, mutates=True)
    p = md.add_parser("select", help="use this model for generation")
    p.add_argument("model_id")
    p.set_defaults(func=cmd_model_select, mutates=True)
    p = md.add_parser("remove", help="remove (deletes only LiteNotebook's own files)")
    p.add_argument("model_id")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_model_remove, mutates=True)

    embeddings = ai_sub.add_parser("embeddings", help="MiniLM for semantic search")
    em = embeddings.add_subparsers(dest="embeddings_command", required=True)
    p = em.add_parser("status")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_embeddings_status)
    p = em.add_parser("install", help="download MiniLM (about 90 MB)")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_embeddings_install, mutates=True)

    ai_sub.add_parser(
        "test", help="start if needed and ask a fixed question"
    ).set_defaults(func=cmd_test, mutates=True)
    ai_sub.add_parser(
        "benchmark", help="measure load, first token, tokens/s"
    ).set_defaults(func=cmd_benchmark, mutates=True)
