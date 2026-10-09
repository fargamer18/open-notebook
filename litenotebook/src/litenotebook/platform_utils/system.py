"""Where LiteNotebook keeps its files, and what machine it is running on.

Default directories (``app_dirs``)::

    Linux   data   $XDG_DATA_HOME/LiteNotebook    (~/.local/share/LiteNotebook)
            config $XDG_CONFIG_HOME/LiteNotebook  (~/.config/LiteNotebook)
            cache  $XDG_CACHE_HOME/LiteNotebook   (~/.cache/LiteNotebook)
    Windows data   %LOCALAPPDATA%\\LiteNotebook
            config %LOCALAPPDATA%\\LiteNotebook
            cache  %LOCALAPPDATA%\\LiteNotebook\\Cache
    macOS   data   ~/Library/Application Support/LiteNotebook (config too)
            cache  ~/Library/Caches/LiteNotebook

XDG variables are used only when they hold absolute paths (the spec says
relative values must be ignored). Nothing here creates directories.

``system_info`` gathers a small, extensible hardware/OS summary without
running shell commands: Linux reads ``/proc`` (this module is the platform
layer, the only place allowed to), Windows uses ``ctypes``/``winreg``, and
anything unknowable is ``None`` rather than a guess.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

APP_NAME = "LiteNotebook"


def os_name(value: str | None = None) -> str:
    """ "windows", "linux", "macos" or "other" (``value`` overrides sys.platform)."""
    raw = value or sys.platform
    if raw.startswith("win"):
        return "windows"
    if raw.startswith("linux"):
        return "linux"
    if raw == "darwin":
        return "macos"
    return "other"


@dataclass(frozen=True)
class AppDirs:
    data: PurePath
    config: PurePath
    cache: PurePath
    log: PurePath | None = None  # default: <data>/logs


def app_dirs(
    env: Mapping[str, str] | None = None,
    *,
    system: str | None = None,
    home: PurePath | None = None,
) -> AppDirs:
    """Platform default directories. ``system``/``home`` let tests simulate
    another OS; for the running OS the result holds concrete ``Path``s."""
    env = os.environ if env is None else env
    kind = os_name(system)
    simulated = system is not None and os_name(system) != os_name()
    if kind == "windows":
        P: type[PurePath] = PureWindowsPath if simulated else Path
        base_home = P(home) if home is not None else P(Path.home())
        local = env.get("LOCALAPPDATA", "").strip()
        root = (
            P(local)
            if local and P(local).is_absolute()
            else base_home / "AppData" / "Local"
        )
        app = root / APP_NAME
        return AppDirs(data=app, config=app, cache=app / "Cache", log=app / "Logs")
    P = PurePosixPath if simulated else Path
    base_home = P(home) if home is not None else P(Path.home())
    if kind == "macos":
        support = base_home / "Library" / "Application Support" / APP_NAME
        return AppDirs(
            support,
            support,
            base_home / "Library" / "Caches" / APP_NAME,
            base_home / "Library" / "Logs" / APP_NAME,
        )

    def xdg(var: str, fallback: PurePath) -> PurePath:
        value = env.get(var, "").strip()
        return P(value) if value and P(value).is_absolute() else fallback

    return AppDirs(
        data=xdg("XDG_DATA_HOME", base_home / ".local" / "share") / APP_NAME,
        config=xdg("XDG_CONFIG_HOME", base_home / ".config") / APP_NAME,
        cache=xdg("XDG_CACHE_HOME", base_home / ".cache") / APP_NAME,
        log=xdg("XDG_STATE_HOME", base_home / ".local" / "state") / APP_NAME / "logs",
    )


def executable_name(base: str, *, system: str | None = None) -> str:
    """``llama-server`` → ``llama-server.exe`` on Windows."""
    if os_name(system) == "windows" and not base.lower().endswith(".exe"):
        return base + ".exe"
    return base


def find_executable(base: str, extra_dirs: tuple[Path, ...] = ()) -> Path | None:
    """An executable by name: explicit directories first, then PATH
    (``shutil.which`` honours PATHEXT on Windows). Never runs a shell."""
    name = executable_name(base)
    for directory in extra_dirs:
        candidate = Path(directory) / name
        if candidate.is_file() and (
            os_name() == "windows" or os.access(candidate, os.X_OK)
        ):
            return candidate
    found = shutil.which(base) or shutil.which(name)
    return Path(found) if found else None


# Helper modules a packaged app can run in a child copy of itself
# (``<app> --internal <module>``); everything else needs its own executable.
INTERNAL_MODULES = frozenset({"litenotebook.platform_utils.supervise"})
INTERNAL_FLAG = "--internal"


def python_module_command(module: str) -> list[str]:
    """How to run ``python -m <module>`` as a child process.

    In a packaged (frozen) build ``sys.executable`` is the app itself: the
    small helpers in ``INTERNAL_MODULES`` run as ``<app> --internal <module>``.
    The embedding worker is a separate executable (``litenotebook.packaging``)
    and anything else is refused rather than starting a second app."""
    if getattr(sys, "frozen", False):
        if module in INTERNAL_MODULES:
            return [sys.executable, INTERNAL_FLAG, module]
        raise RuntimeError(f"{module} cannot be started from a packaged build")
    return [sys.executable, "-m", module]


def open_browser(url: str) -> bool:
    """Open a URL in the default browser (no xdg-open/start/open subprocess)."""
    import webbrowser

    try:
        return webbrowser.open(url, new=2)
    except webbrowser.Error:
        return False


def open_folder(path: Path) -> bool:
    """Show a folder in the file manager. The path is passed as a file:// URL
    to the standard ``webbrowser`` dispatcher: no shell, no command string."""
    import webbrowser

    try:
        return path.is_dir() and webbrowser.open(path.resolve().as_uri())
    except (webbrowser.Error, ValueError, OSError):
        return False


# --- hardware / OS summary ---------------------------------------------------------


@dataclass
class SystemInfo:
    """Extended in Phase 12 (GPU, accelerators) — keep fields optional."""

    os: str
    os_release: str
    architecture: str
    python_version: str
    python_implementation: str
    cpu_name: str | None
    logical_cpus: int | None
    physical_cpus: int | None
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    disk_free_bytes: int | None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _linux_cpu() -> tuple[str | None, int | None]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    name = None
    cores: set[tuple[str, str]] = set()
    physical, core = "0", None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and name is None:
            name = value
        elif key == "physical id":
            physical = value
        elif key == "core id":
            core = value
        elif not key and core is not None:
            cores.add((physical, core))
            core = None
    if core is not None:
        cores.add((physical, core))
    return name, (len(cores) or None)


def _linux_memory() -> tuple[int | None, int | None]:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0]) * 1024
    return values.get("MemTotal"), values.get("MemAvailable")


def _windows_memory() -> tuple[int | None, int | None]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None, None
    return (int(status.ullTotalPhys), int(status.ullAvailPhys)) if ok else (None, None)


def _windows_cpu_name() -> str | None:
    try:
        import winreg  # type: ignore[import-not-found,unused-ignore]

        with winreg.OpenKey(  # type: ignore[attr-defined,unused-ignore]
            winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined,unused-ignore]
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")  # type: ignore[attr-defined,unused-ignore]
            return str(value).strip() or None
    except (ImportError, OSError):
        return platform.processor() or None


def _posix_total_memory() -> int | None:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def system_info(disk_path: Path | None = None) -> SystemInfo:
    kind = os_name()
    cpu_name: str | None = None
    physical: int | None = None
    total: int | None = None
    available: int | None = None
    if kind == "linux":
        cpu_name, physical = _linux_cpu()
        total, available = _linux_memory()
    elif kind == "windows":
        cpu_name = _windows_cpu_name()
        total, available = _windows_memory()
    else:
        cpu_name = platform.processor() or None
    if total is None and kind != "windows":
        total = _posix_total_memory()
    disk_free = None
    if disk_path is not None:
        probe = Path(disk_path)
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            disk_free = shutil.disk_usage(probe).free
        except OSError:
            disk_free = None
    return SystemInfo(
        os=platform.system() or kind,
        os_release=platform.release(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        cpu_name=cpu_name,
        logical_cpus=os.cpu_count(),
        physical_cpus=physical,
        total_ram_bytes=total,
        available_ram_bytes=available,
        disk_free_bytes=disk_free,
    )


def process_rss_bytes(pid: int) -> int | None:
    """Resident memory of a process LiteNotebook started (None if unknown)."""
    kind = os_name()
    if kind == "linux":
        try:
            for line in (
                Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
            ):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None
    if kind == "windows":

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                *[
                    (n, ctypes.c_size_t)
                    for n in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ],
            ]

        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
            if not handle:
                return None
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return int(counters.WorkingSetSize)
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError):
            return None
    return None


def with_library_dir(
    env: Mapping[str, str], library_dir: Path, *, system: str | None = None
) -> dict[str, str]:
    """``env`` plus ``library_dir`` on the shared-library search path of a
    child process: ``PATH`` on Windows (DLLs next to an exe's ``lib``
    folder), ``LD_LIBRARY_PATH`` on Linux, ``DYLD_LIBRARY_PATH`` on macOS."""
    kind = os_name(system)
    key = {"windows": "PATH", "macos": "DYLD_LIBRARY_PATH"}.get(kind, "LD_LIBRARY_PATH")
    out = dict(env)
    current = out.get(key, "")
    sep = ";" if kind == "windows" else ":"
    out[key] = str(library_dir) + (sep + current if current else "")
    return out


def machine_arch() -> str:
    """This computer's CPU architecture as reported by the OS (e.g. x86_64)."""
    return platform.machine() or "unknown"


def show_error_dialog(title: str, message: str, *, system: str | None = None) -> bool:
    """A native error box where one exists without extra dependencies
    (Windows MessageBoxW). Returns False elsewhere; callers then fall back
    to a page in the browser and the log file."""
    if os_name(system) != "windows":
        return False
    try:
        user32 = ctypes.WinDLL("user32")  # type: ignore[attr-defined]
        user32.MessageBoxW(None, message, title, 0x10)  # MB_ICONERROR
        return True
    except (AttributeError, OSError):
        return False
