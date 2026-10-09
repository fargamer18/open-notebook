"""llama.cpp runtimes: managed installs, existing executables, probing.

Managed install lifecycle (``install_runtime``)::

    disk preflight → download each asset (.part, resumable, SHA-256 verified)
    → safe extraction into <runtime>/.staging-<id> → expected executable
    exists inside staging → it runs (``--version``) → rename into
    <runtime>/<id> (the previous copy is kept until the new one is in place)
    → write litenotebook-runtime.json

Nothing is executed before its archive's checksum has been verified, and an
install is only recorded once every step succeeded. An existing llama-server
the user already has is probed with ``--version`` / ``--list-devices`` (no
shell, bounded time) and used in place — never copied unless they choose a
managed install.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from litenotebook.ai.catalog import RuntimeEntry
from litenotebook.ai.download import Downloader, Progress, check_space
from litenotebook.ai.extract import extract
from litenotebook.config import Settings
from litenotebook.platform_utils.filesystem import is_within, remove_tree, rename_dir
from litenotebook.platform_utils.processes import creation_flags
from litenotebook.platform_utils.system import executable_name, find_executable, os_name

METADATA = "litenotebook-runtime.json"
_VERSION = re.compile(r"version:\s*(\S+)\s*\(build\s*(\d+)(?:,\s*commit\s*(\w+))?")
_DEVICE = re.compile(
    r"^\s+(CUDA\d+|Vulkan\d+|ROCm\d+|SYCL\d+|Metal\d*)\s*:\s*(.+)$", re.M
)
PROBE_TIMEOUT_S = 20.0


class RuntimeError_(RuntimeError):
    """A runtime problem with a message safe to show."""


@dataclass
class RuntimeProbe:
    ok: bool
    executable: str
    version: str | None = None
    build: int | None = None
    devices: list[str] | None = None
    backend: str = "cpu"
    error: str | None = None


@dataclass
class InstalledRuntime:
    id: str
    version: str
    backend: str
    os: str
    arch: str
    executable: str  # absolute
    path: str
    installed_at: float
    managed: bool = True


def runtime_root(settings: Settings) -> Path:
    return settings.data_dir / "runtime" / "llama.cpp"


def downloads_dir(settings: Settings) -> Path:
    return settings.data_dir / "tmp" / "downloads"


def _run(argv: list[str], timeout: float) -> tuple[int, str]:
    result = subprocess.run(  # noqa: S603 - argument list, no shell
        argv,
        capture_output=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        creationflags=creation_flags(),
    )
    text = (result.stdout + b"\n" + result.stderr).decode("utf-8", "replace")
    return result.returncode, text


def probe(executable: Path | str, *, list_devices: bool = True) -> RuntimeProbe:
    """Is this a working llama-server? Never through a shell; bounded time.
    ``list_devices`` asks the binary which GPUs it can use (briefly touches
    the GPU driver, allocating nothing lasting)."""
    path = Path(executable)
    shown = str(path)
    if not path.is_file():
        return RuntimeProbe(False, shown, error="The file does not exist")
    expected = executable_name("llama-server")
    if path.name.lower() != expected.lower():
        return RuntimeProbe(False, shown, error=f"Choose the {expected} program")
    try:
        code, out = _run([str(path), "--version"], PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return RuntimeProbe(False, shown, error="It did not answer --version in time")
    except OSError as e:
        return RuntimeProbe(
            False, shown, error=f"It could not be started ({e.strerror or e})"
        )
    match = _VERSION.search(out)
    if code != 0 or not match:
        return RuntimeProbe(
            False,
            shown,
            error="It does not look like llama-server (no version reported)",
        )
    devices: list[str] = []
    if list_devices:
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            _, listing = _run([str(path), "--list-devices"], PROBE_TIMEOUT_S)
            devices = [
                f"{m.group(1)}: {m.group(2).strip()}" for m in _DEVICE.finditer(listing)
            ]
    backend = "cuda" if any(d.startswith("CUDA") for d in devices) else "cpu"
    return RuntimeProbe(
        True,
        shown,
        version=match.group(1),
        build=int(match.group(2)),
        devices=devices,
        backend=backend,
    )


_DISCOVERED: dict[tuple[str, int, int], RuntimeProbe] = {}


def discover_existing() -> list[RuntimeProbe]:
    """llama-server on PATH (``shutil.which``; PATHEXT on Windows).

    For page views: ``--version`` only (no ``--list-devices``, which wakes the
    GPU driver), remembered per executable until it changes on disk."""
    found = find_executable("llama-server")
    if not found:
        return []
    try:
        st = Path(found).stat()
    except OSError:
        return []
    key = (str(found), st.st_size, st.st_mtime_ns)
    if key not in _DISCOVERED:
        _DISCOVERED[key] = probe(found, list_devices=False)
    return [_DISCOVERED[key]]


def list_installed(settings: Settings) -> list[InstalledRuntime]:
    root = runtime_root(settings)
    out: list[InstalledRuntime] = []
    if not root.is_dir():
        return out
    for meta in sorted(root.glob(f"*/{METADATA}")):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            exe = meta.parent / data["executable"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not exe.is_file() or not is_within(exe, meta.parent):
            continue
        out.append(
            InstalledRuntime(
                id=data["id"],
                version=data["version"],
                backend=data["backend"],
                os=data["os"],
                arch=data["arch"],
                executable=str(exe),
                path=str(meta.parent),
                installed_at=float(data.get("installed_at", 0)),
            )
        )
    return out


def install_runtime(
    settings: Settings,
    entry: RuntimeEntry,
    *,
    downloader: Downloader,
    progress: Callable[[str, Progress | None], None] | None = None,
    cancel: threading.Event | None = None,
    check_runs: bool = True,
) -> InstalledRuntime:
    def say(stage: str, value: Progress | None = None) -> None:
        if progress is not None:
            progress(stage, value)

    if entry.os != os_name():
        raise RuntimeError_(f"{entry.id} is for {entry.os}, not this computer")
    root = runtime_root(settings)
    root.mkdir(parents=True, exist_ok=True)
    tmp = downloads_dir(settings)
    # archives + extracted copy, then the reserve
    check_space(root, entry.download_size * 3)
    archives = []
    for i, asset in enumerate(entry.assets, 1):
        name = asset.url.rsplit("/", 1)[1]
        label = f"Downloading {name}" + (
            f" ({i} of {len(entry.assets)})" if len(entry.assets) > 1 else ""
        )
        say(label, None)

        def report(p: Progress, label: str = label) -> None:
            say(label, p)

        fetched = downloader.fetch(
            asset.url,
            tmp / name,
            sha256=asset.sha256,
            size=asset.size,
            progress=report,
            cancel=cancel,
        )
        archives.append((fetched.path, asset.format))
    staging = root / f".staging-{uuid.uuid4().hex[:12]}"
    try:
        say("Extracting…", None)
        for archive, fmt in archives:
            extract(archive, fmt, staging)
        exe = staging / entry.executable
        if not exe.is_file() or not is_within(exe, staging):
            raise RuntimeError_(
                f"The runtime archive does not contain {entry.executable}; nothing was "
                "installed"
            )
        if check_runs:
            say("Checking the runtime…", None)
            result = probe(exe, list_devices=False)
            if not result.ok:
                raise RuntimeError_(
                    f"The downloaded runtime does not run here: {result.error}"
                )
        (staging / METADATA).write_text(
            json.dumps(
                {
                    "id": entry.id,
                    "version": entry.version,
                    "backend": entry.backend,
                    "os": entry.os,
                    "arch": entry.arch,
                    "executable": entry.executable,
                    "installed_at": time.time(),
                    "assets": [a.sha256 for a in entry.assets],
                    "license": entry.license,
                    "release_url": entry.release_url,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        final = root / entry.id
        old = None
        if final.exists():  # repair/reinstall: keep the old copy until the swap
            old = root / f".old-{uuid.uuid4().hex[:12]}"
            rename_dir(final, old)
        try:
            rename_dir(staging, final)
        except BaseException:
            if old is not None:
                rename_dir(old, final)
            raise
        if old is not None:
            with contextlib.suppress(OSError):
                remove_tree(old)
    except BaseException:
        with contextlib.suppress(OSError):
            remove_tree(staging)
        raise
    finally:
        for archive, _ in archives:
            with contextlib.suppress(OSError):
                archive.unlink(missing_ok=True)
    say("Installed", None)
    installed = [r for r in list_installed(settings) if r.id == entry.id]
    if not installed:
        raise RuntimeError_(
            "The runtime was installed but could not be found afterwards"
        )
    return installed[0]


def as_dict(item: InstalledRuntime | RuntimeProbe) -> dict[str, object]:
    return asdict(item)
