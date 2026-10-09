"""Hardware profile for setup and model recommendations.

Built from ``platform_utils.system.system_info`` and
``platform_utils.gpu.detect_gpus``: no torch, no MiniLM, no llama.cpp. Values
that cannot be measured stay ``None`` ("Unknown") and are never invented;
recommendations then fall back to conservative choices.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from litenotebook.ai.catalog import normalize_arch
from litenotebook.platform_utils.gpu import GPU, detect_gpus
from litenotebook.platform_utils.system import os_name, system_info


@dataclass
class GPUInfo:
    vendor: str
    name: str
    vram_bytes: int | None
    vram_free_bytes: int | None
    integrated: bool
    confidence: str  # confirmed | estimated | unknown
    source: str
    driver_version: str | None
    backend_candidates: list[str] = field(default_factory=list)

    @classmethod
    def from_gpu(cls, gpu: GPU) -> GPUInfo:
        candidates = ["cpu"]
        if gpu.vendor == "nvidia" and not gpu.integrated:
            candidates.insert(0, "cuda")
        return cls(
            gpu.vendor,
            gpu.name,
            gpu.vram_bytes,
            gpu.vram_free_bytes,
            gpu.integrated,
            gpu.confidence,
            gpu.source,
            gpu.driver_version,
            candidates,
        )


@dataclass
class HardwareProfile:
    platform: str  # linux | windows | macos | other
    os_release: str
    architecture: str  # x86_64 | arm64 | …
    cpu: str | None
    physical_cores: int | None
    logical_cores: int | None
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    free_disk_bytes: int | None
    gpus: list[GPUInfo] = field(default_factory=list)
    scanned_at: float = 0.0
    scan_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareProfile:
        gpus = [GPUInfo(**g) for g in data.get("gpus") or []]
        fields = {k: data.get(k) for k in cls.__dataclass_fields__ if k != "gpus"}
        return cls(**fields, gpus=gpus)  # type: ignore[arg-type]

    @property
    def best_gpu(self) -> GPUInfo | None:
        """The discrete GPU with the most measurable VRAM (one GPU is used)."""
        usable = [g for g in self.gpus if not g.integrated]
        if not usable:
            return None
        return max(usable, key=lambda g: g.vram_bytes or 0)


def scan_hardware(disk_path: Path) -> HardwareProfile:
    started = time.perf_counter()
    info = system_info(disk_path)
    try:
        gpus = [GPUInfo.from_gpu(g) for g in detect_gpus()]
    except Exception:  # detection is best effort; CPU setup always works
        gpus = []
    return HardwareProfile(
        platform=os_name(),
        os_release=f"{info.os} {info.os_release}".strip(),
        architecture=normalize_arch(info.architecture),
        cpu=info.cpu_name,
        physical_cores=info.physical_cpus,
        logical_cores=info.logical_cpus,
        total_ram_bytes=info.total_ram_bytes,
        available_ram_bytes=info.available_ram_bytes,
        free_disk_bytes=info.disk_free_bytes,
        gpus=gpus,
        scanned_at=time.time(),
        scan_ms=round((time.perf_counter() - started) * 1000, 1),
    )
