"""GPU detection without torch, drivers' CLIs or a shell.

Sources, best first:

1. **NVML** (NVIDIA's management library, shipped with the driver:
   ``libnvidia-ml.so.1`` / ``nvml.dll``) through ``ctypes`` — name, total and
   free VRAM, driver version. Confidence ``confirmed``.
2. **Linux sysfs** (``/sys/class/drm/card*/device``): PCI vendor/device ids,
   and ``mem_info_vram_total`` where the driver exposes it (amdgpu) →
   ``confirmed``; otherwise VRAM ``unknown``.
3. **Windows registry** display-adapter keys: ``DriverDesc`` and
   ``HardwareInformation.qwMemorySize`` → ``estimated`` (the value is what the
   driver reports; it is not measured).

Integrated GPUs share system RAM; their VRAM is reported as unknown, never
guessed. Every probe is best effort: failure returns no GPUs, and CPU-only
setup always remains possible.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litenotebook.platform_utils.system import os_name

log = logging.getLogger(__name__)

VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel", 0x106B: "apple"}
CONFIRMED, ESTIMATED, UNKNOWN = "confirmed", "estimated", "unknown"


@dataclass
class GPU:
    vendor: str  # nvidia | amd | intel | other
    name: str
    vram_bytes: int | None = None
    vram_free_bytes: int | None = None
    integrated: bool = False
    confidence: str = UNKNOWN  # of the VRAM figure
    source: str = ""
    driver_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --- NVML ---------------------------------------------------------------------------


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _load_nvml() -> Any:
    names = (
        ["nvml.dll", r"C:\Windows\System32\nvml.dll"]
        if os_name() == "windows"
        else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    )
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def nvml_gpus() -> list[GPU]:
    lib = _load_nvml()
    if lib is None:
        return []
    gpus: list[GPU] = []
    try:
        if lib.nvmlInit_v2() != 0:
            return []
    except AttributeError:
        return []
    try:
        driver = ctypes.create_string_buffer(96)
        driver_version = None
        if lib.nvmlSystemGetDriverVersion(driver, 96) == 0:
            driver_version = driver.value.decode("ascii", "replace")
        count = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            return []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                continue
            name = ctypes.create_string_buffer(96)
            lib.nvmlDeviceGetName(handle, name, 96)
            memory = _NvmlMemory()
            ok = lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0
            gpus.append(
                GPU(
                    vendor="nvidia",
                    name=name.value.decode("utf-8", "replace") or "NVIDIA GPU",
                    vram_bytes=int(memory.total) if ok else None,
                    vram_free_bytes=int(memory.free) if ok else None,
                    confidence=CONFIRMED if ok else UNKNOWN,
                    source="nvml",
                    driver_version=driver_version,
                )
            )
    except (AttributeError, OSError, ValueError) as e:
        log.debug("NVML query failed: %s", e)
    finally:
        with contextlib.suppress(AttributeError, OSError):
            lib.nvmlShutdown()
    return gpus


# --- Linux sysfs --------------------------------------------------------------------


def _read_hex(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip(), 16)
    except (OSError, ValueError):
        return None


def sysfs_gpus(root: Path = Path("/sys/class/drm")) -> list[GPU]:
    gpus: list[GPU] = []
    seen: set[str] = set()
    try:
        cards = sorted(
            p for p in root.iterdir() if p.name.startswith("card") and "-" not in p.name
        )
    except OSError:
        return []
    for card in cards:
        device = card / "device"
        try:
            pci = device.resolve().name
        except OSError:
            continue
        if pci in seen:
            continue
        seen.add(pci)
        vendor_id = _read_hex(device / "vendor")
        if vendor_id is None:
            continue
        cls = _read_hex(device / "class")
        if cls is not None and (cls >> 16) != 0x03:  # not a display controller
            continue
        device_id = _read_hex(device / "device")
        vendor = VENDORS.get(vendor_id, "other")
        vram = None
        with contextlib.suppress(OSError, ValueError):
            vram = int(
                (device / "mem_info_vram_total").read_text(encoding="ascii").strip()
            )
        vram_free = None
        with contextlib.suppress(OSError, ValueError):
            used = int(
                (device / "mem_info_vram_used").read_text(encoding="ascii").strip()
            )
            vram_free = vram - used if vram is not None else None
        # Intel GPUs and AMD APUs with tiny carve-outs share system memory
        integrated = vendor == "intel" or (
            vendor == "amd" and vram is not None and vram < 1 << 30
        )
        gpus.append(
            GPU(
                vendor=vendor,
                name=f"{vendor.upper() if vendor != 'other' else 'GPU'} device "
                f"{device_id:#06x}"
                if device_id is not None
                else f"{vendor} GPU",
                vram_bytes=None if integrated else vram,
                vram_free_bytes=None if integrated else vram_free,
                integrated=integrated,
                confidence=CONFIRMED
                if vram is not None and not integrated
                else UNKNOWN,
                source="sysfs",
                extra={"pci": pci},
            )
        )
    return gpus


# --- Windows registry ---------------------------------------------------------------


_DISPLAY_CLASS = (
    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
)


def windows_registry_gpus() -> list[GPU]:
    try:
        import winreg  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return []
    gpus: list[GPU] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)  # type: ignore[attr-defined,unused-ignore]
    except OSError:
        return []
    with root:
        for index in range(32):
            try:
                sub = winreg.EnumKey(root, index)  # type: ignore[attr-defined,unused-ignore]
            except OSError:
                break
            if not sub.isdigit():
                continue
            try:
                key = winreg.OpenKey(root, sub)  # type: ignore[attr-defined,unused-ignore]
            except OSError:
                continue
            with key:

                def value(name: str, key: Any = key) -> Any:
                    try:
                        return winreg.QueryValueEx(key, name)[0]  # type: ignore[attr-defined,unused-ignore]
                    except OSError:
                        return None

                name = value("DriverDesc")
                if not name:
                    continue
                memory = value("HardwareInformation.qwMemorySize") or value(
                    "HardwareInformation.MemorySize"
                )
                if isinstance(memory, bytes):
                    memory = int.from_bytes(memory[:8], "little")
                lower = str(name).lower()
                vendor = next(
                    (
                        v
                        for v in ("nvidia", "amd", "intel")
                        if v in lower or (v == "amd" and "radeon" in lower)
                    ),
                    "other",
                )
                integrated = vendor == "intel" and "arc" not in lower
                gpus.append(
                    GPU(
                        vendor=vendor,
                        name=str(name),
                        vram_bytes=None if integrated or not memory else int(memory),
                        integrated=integrated,
                        confidence=ESTIMATED if memory and not integrated else UNKNOWN,
                        source="registry",
                        driver_version=value("DriverVersion"),
                    )
                )
    return gpus


def detect_gpus() -> list[GPU]:
    """All display adapters found; NVML data replaces weaker entries for NVIDIA."""
    nvidia = nvml_gpus()
    others: list[GPU] = []
    kind = os_name()
    if kind == "linux":
        others = sysfs_gpus()
    elif kind == "windows":
        others = windows_registry_gpus()
    if nvidia:
        others = [g for g in others if g.vendor != "nvidia"]
    return [*nvidia, *others]
