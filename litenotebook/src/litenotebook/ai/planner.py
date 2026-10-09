"""Model recommendations and conservative llama.cpp launch plans.

Decisions use capabilities (RAM, VRAM, backend, disk), never GPU model names.
Memory is estimated from the model file size (weights) plus the KV cache for
the chosen context (layers × KV heads × head dim × 2 × bytes per element)
plus a fixed runtime overhead, and must fit with a safety margin:

* GPU: 10 % of VRAM or 384 MiB (whichever is larger) stays free; when free
  VRAM can be measured right now (another program may be using the GPU) the
  plan uses that instead of the total.
* RAM: 3 GiB for the OS, browser and LiteNotebook, plus 1 GiB when the
  MiniLM worker may be running.

Unknown VRAM is treated as "no usable GPU" for sizing. Stability over speed:
one parallel slot, contexts of 8192 → 4096 → 2048, q8_0 KV cache with flash
attention, then safer fallbacks. Nothing here predicts tokens per second.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from litenotebook.ai.catalog import ModelEntry
from litenotebook.ai.hardware import GPUInfo, HardwareProfile

MiB = 1024**2
GiB = 1024**3
GPU_OVERHEAD = 512 * MiB  # CUDA context + compute buffers
CPU_OVERHEAD = 384 * MiB
RAM_RESERVE = 3 * GiB
MINILM_RESERVE = 1 * GiB
DISK_RESERVE = 2 * GiB
KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
CONTEXTS = (8192, 4096, 2048)
RATINGS = ("recommended", "good", "slow", "not_recommended")


@dataclass
class LaunchPlan:
    context: int
    gpu_layers: int  # 0 = CPU only; 999 = all layers
    kv_type: str
    flash_attn: str  # on | auto
    threads: int
    parallel: int
    placement: str  # gpu | partial | cpu
    est_vram_bytes: int
    est_ram_bytes: int
    explanation: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fit:
    model_id: str
    rating: str
    reasons: list[str]
    plan: LaunchPlan | None
    recommended: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["plan"] = self.plan.as_dict() if self.plan else None
        return d


def kv_bytes(arch: dict[str, int], context: int, kv_type: str) -> int:
    elements = 2 * arch["layers"] * arch["kv_heads"] * arch["head_dim"] * context
    return int(elements * KV_BYTES[kv_type])


def _gb(n: int | float) -> str:
    return f"{n / GiB:.1f} GiB"


def threads_for(hw: HardwareProfile) -> int:
    if hw.physical_cores:
        return max(1, min(hw.physical_cores, 16))
    if hw.logical_cores:
        return max(1, min(hw.logical_cores // 2 or 1, 16))
    return 4


def usable_vram(gpu: GPUInfo | None, *, live: bool = False) -> int:
    """VRAM a model may use, after the margin. 0 when unknown."""
    if gpu is None or gpu.integrated or not gpu.vram_bytes:
        return 0
    if gpu.confidence == "unknown":
        return 0
    base = (
        gpu.vram_free_bytes
        if live and gpu.vram_free_bytes is not None
        else gpu.vram_bytes
    )
    margin = max(int(gpu.vram_bytes * 0.10), 384 * MiB)
    return max(0, base - margin)


def usable_ram(hw: HardwareProfile, *, minilm: bool = True) -> int:
    """RAM a model may use: the smaller of total-minus-reserve and what is free."""
    if not hw.total_ram_bytes:
        return 0
    reserve = RAM_RESERVE + (MINILM_RESERVE if minilm else 0)
    total_room = hw.total_ram_bytes - reserve
    if hw.available_ram_bytes is not None:
        total_room = min(total_room, hw.available_ram_bytes - MINILM_RESERVE * minilm)
    return max(0, total_room)


def plan_launch(
    model: ModelEntry,
    hw: HardwareProfile,
    *,
    backend: str,
    target_context: int = 8192,
    live_vram: bool = False,
    minilm: bool = True,
) -> LaunchPlan | None:
    """The most generous plan that still fits with margin, or None."""
    arch, weights = model.architecture, model.size
    threads = threads_for(hw)
    contexts = [c for c in CONTEXTS if c <= target_context] or [2048]
    gpu = hw.best_gpu if backend == "cuda" else None
    vram = usable_vram(gpu, live=live_vram)
    ram = usable_ram(hw, minilm=minilm)
    if backend == "cuda" and vram > 0:
        for ctx in contexts:
            need = weights + kv_bytes(arch, ctx, "q8_0") + GPU_OVERHEAD
            if need <= vram:
                return LaunchPlan(
                    ctx,
                    999,
                    "q8_0",
                    "on",
                    threads,
                    1,
                    "gpu",
                    need,
                    CPU_OVERHEAD,
                    [
                        f"all layers on the GPU ({_gb(need)} of {_gb(vram)} usable "
                        "VRAM)",
                        f"{ctx:,}-token context with a compact (q8_0) KV cache",
                    ],
                )
        ctx = 4096 if 4096 in contexts else contexts[-1]
        per_layer = weights / arch["layers"]
        room = vram - kv_bytes(arch, ctx, "q8_0") - GPU_OVERHEAD
        layers = int(room // per_layer) if room > 0 else 0
        cpu_part = int(weights - layers * per_layer) + CPU_OVERHEAD
        if layers >= max(4, arch["layers"] // 4) and cpu_part <= ram:
            return LaunchPlan(
                ctx,
                layers,
                "q8_0",
                "on",
                threads,
                1,
                "partial",
                int(vram),
                cpu_part,
                [
                    f"{layers} of {arch['layers']} layers on the GPU, "
                    "the rest on the CPU",
                    f"{ctx:,}-token context",
                ],
            )
    for ctx in contexts:
        need = weights + kv_bytes(arch, ctx, "q8_0") + CPU_OVERHEAD
        if need <= ram:
            why = (
                "the GPU cannot be used"
                if backend == "cpu"
                else "the model does not fit in GPU memory"
            )
            return LaunchPlan(
                ctx,
                0,
                "q8_0",
                "on",
                threads,
                1,
                "cpu",
                0,
                need,
                [
                    f"runs on the CPU ({why}); needs about {_gb(need)} of "
                    f"{_gb(ram)} usable RAM",
                    f"{ctx:,}-token context, {threads} CPU threads",
                ],
            )
    return None


def fallback_plans(
    model: ModelEntry, first: LaunchPlan
) -> list[tuple[LaunchPlan, str]]:
    """Safer plans to try in order after ``first`` fails (e.g. out of memory)."""
    out: list[tuple[LaunchPlan, str]] = []
    current = first
    if current.context > 2048:
        smaller = max(2048, current.context // 2)
        current = LaunchPlan(
            **{**current.as_dict(), "context": smaller, "explanation": []}
        )
        out.append((current, f"reduced the context to {smaller:,} tokens"))
    if current.gpu_layers:
        layers = model.architecture["layers"]
        half = (layers if current.gpu_layers >= layers else current.gpu_layers) // 2
        if half >= 4:
            current = LaunchPlan(
                **{
                    **current.as_dict(),
                    "gpu_layers": half,
                    "placement": "partial",
                    "explanation": [],
                }
            )
            out.append(
                (
                    current,
                    f"moved part of the model to the CPU ({half} layers on the GPU)",
                )
            )
        current = LaunchPlan(
            **{
                **current.as_dict(),
                "gpu_layers": 0,
                "placement": "cpu",
                "explanation": [],
            }
        )
        out.append((current, "ran the model on the CPU only"))
    if current.kv_type != "f16" or current.flash_attn != "auto":
        current = LaunchPlan(
            **{
                **current.as_dict(),
                "kv_type": "f16",
                "flash_attn": "auto",
                "context": min(current.context, 2048),
                "explanation": [],
            }
        )
        out.append(
            (
                current,
                "used the most compatible KV-cache settings with a 2,048-token context",
            )
        )
    return out


def rate(
    model: ModelEntry,
    hw: HardwareProfile,
    backends: list[str],
    *,
    installed: bool = False,
    minilm: bool = True,
) -> Fit:
    reasons: list[str] = []
    disk_need = 0 if installed else model.size + DISK_RESERVE
    if hw.free_disk_bytes is not None and disk_need > hw.free_disk_bytes:
        return Fit(
            model.id,
            "not_recommended",
            [
                f"needs {_gb(model.size)} of disk space (plus "
                f"{_gb(DISK_RESERVE)} kept free); "
                f"{_gb(hw.free_disk_bytes)} free"
            ],
            None,
        )
    plan = None
    for backend in backends:
        plan = plan_launch(model, hw, backend=backend, minilm=minilm)
        if plan is not None:
            break
    if plan is None:
        return Fit(
            model.id,
            "not_recommended",
            ["not enough memory for this model with a safe margin"],
            None,
        )
    reasons += plan.explanation
    if hw.total_ram_bytes:
        reasons.append(f"{_gb(hw.total_ram_bytes)} RAM in this computer")
    if plan.placement == "gpu":
        rating = "recommended" if plan.context >= 4096 else "good"
    elif plan.placement == "partial":
        rating = "good" if model.size_class != "large" else "slow"
    else:
        cores = hw.physical_cores or (hw.logical_cores or 0) // 2
        if model.size_class == "light":
            rating = "good"
        elif model.size_class == "balanced" and cores >= 6 and plan.context >= 4096:
            rating = "good"
        else:
            rating = "slow"
        if rating == "slow":
            reasons.append("runs on the CPU, so answers will be slow")
    gpu = hw.best_gpu
    if gpu is not None and not gpu.integrated and gpu.confidence == "unknown":
        reasons.append(
            "GPU memory could not be measured, so CPU settings were chosen to be safe"
        )
    return Fit(model.id, rating, reasons, plan)


def recommend(
    models: list[ModelEntry],
    hw: HardwareProfile,
    backends: list[str],
    *,
    installed_ids: frozenset[str] = frozenset(),
    minilm: bool = True,
) -> list[Fit]:
    """Every model rated; the best-quality model rated "recommended" or
    "good" (largest first) is marked as the recommendation."""
    order = {"large": 0, "balanced": 1, "light": 2}
    fits = [
        rate(m, hw, backends, installed=m.id in installed_ids, minilm=minilm)
        for m in models
    ]
    by_id = {m.id: m for m in models}
    candidates = sorted(
        (f for f in fits if f.rating in ("recommended", "good")),
        key=lambda f: (
            RATINGS.index(f.rating),
            order.get(by_id[f.model_id].size_class, 3),
        ),
    )
    if not candidates:
        candidates = sorted(
            (f for f in fits if f.rating == "slow"),
            key=lambda f: -order.get(by_id[f.model_id].size_class, 3),
        )
    if candidates:
        candidates[0].recommended = True
    return sorted(fits, key=lambda f: (not f.recommended, RATINGS.index(f.rating)))


def available_backends(hw: HardwareProfile, runtime_backends: list[str]) -> list[str]:
    """Backends worth trying, best first: CUDA only with a discrete NVIDIA GPU
    whose memory was measured and a runtime that supports it."""
    out = []
    gpu = hw.best_gpu
    if (
        "cuda" in runtime_backends
        and gpu is not None
        and gpu.vendor == "nvidia"
        and gpu.confidence in ("confirmed", "estimated")
        and gpu.vram_bytes
    ):
        out.append("cuda")
    out.append("cpu")
    return out


def needs_disk(size: int) -> int:
    return size + DISK_RESERVE


def gpu_layers_flag(plan: LaunchPlan) -> str:
    return str(plan.gpu_layers if plan.gpu_layers < 999 else 999)


def estimate_memory(model: ModelEntry, plan: LaunchPlan) -> int:
    return int(
        model.size
        + kv_bytes(model.architecture, plan.context, plan.kv_type)
        + (GPU_OVERHEAD if plan.gpu_layers else CPU_OVERHEAD)
    )


__all__ = [
    "Fit",
    "LaunchPlan",
    "available_backends",
    "estimate_memory",
    "fallback_plans",
    "gpu_layers_flag",
    "kv_bytes",
    "needs_disk",
    "plan_launch",
    "rate",
    "recommend",
]
