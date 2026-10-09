"""Phase 12: hardware detection, pinned catalogs and conservative recommendations.

Recommendations come from measured numbers only (RAM, VRAM and how sure we
are of it, disk, cores) — never from a GPU's name.
"""

import json
import sys
from pathlib import Path

import pytest

from litenotebook.ai import catalog as cat
from litenotebook.ai.hardware import GPUInfo, HardwareProfile, scan_hardware
from litenotebook.ai.planner import (
    available_backends,
    fallback_plans,
    kv_bytes,
    plan_launch,
    recommend,
    usable_vram,
)
from litenotebook.platform_utils import gpu as gpu_mod

GiB = 1024**3
MODELS = cat.load_models()
QWEN4B = cat.model("qwen3-4b-q4_k_m")


def hw(ram_gb=16, free_gb=None, cores=8, disk_gb=200, gpus=()):
    return HardwareProfile(
        platform="linux",
        os_release="Linux test",
        architecture="x86_64",
        cpu="Test CPU",
        physical_cores=cores,
        logical_cores=cores * 2,
        total_ram_bytes=int(ram_gb * GiB),
        available_ram_bytes=int(
            (free_gb if free_gb is not None else ram_gb * 0.8) * GiB
        ),
        free_disk_bytes=int(disk_gb * GiB),
        gpus=list(gpus),
    )


def nvidia(vram_gb, *, name="Some NVIDIA GPU", confidence="confirmed", free_gb=None):
    vram = int(vram_gb * GiB) if vram_gb else None
    return GPUInfo(
        "nvidia",
        name,
        vram,
        int(free_gb * GiB) if free_gb is not None else vram,
        False,
        confidence,
        "nvml",
        "570.0",
        ["cuda", "cpu"],
    )


def integrated():
    return GPUInfo(
        "amd", "AMD device 0x1636", None, None, True, "unknown", "sysfs", None, ["cpu"]
    )


def pick(profile, backends=None):
    fits = recommend(
        MODELS, profile, backends or available_backends(profile, ["cpu", "cuda"])
    )
    best = next((f for f in fits if f.recommended), None)
    return best, {f.model_id: f for f in fits}


# --- catalogs -------------------------------------------------------------------------


def test_shipped_catalogs_load_and_are_pinned():
    runtimes, models, embeddings = (
        cat.load_runtimes(),
        cat.load_models(),
        cat.load_embeddings(),
    )
    assert {(r.os, r.backend) for r in runtimes} >= {
        ("linux", "cpu"),
        ("windows", "cpu"),
    }
    for r in runtimes:
        assert r.version == "b11146" and r.assets
        assert all(cat.host_allowed(a.url) and len(a.sha256) == 64 for a in r.assets)
        assert r.executable.endswith(
            "llama-server" + (".exe" if r.os == "windows" else "")
        )
    assert {m.id for m in models} == {
        "qwen3-1.7b-q8_0",
        "qwen3-4b-q4_k_m",
        "qwen3-8b-q4_k_m",
    }
    for m in models:
        assert (
            "/resolve/" in m.url and m.revision in m.url
        )  # a fixed revision, not "main"
        assert m.license == "Apache-2.0" and m.size > 10**9
    assert embeddings[0].revision and len(embeddings[0].files) == 10


def _with(kind, mutate):
    raw = json.loads(
        (Path(cat.__file__).parent / "catalog" / f"{kind}.json").read_text(
            encoding="utf-8"
        )
    )
    mutate(raw)
    return json.dumps(raw)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: d["models"][0].update(url="https://evil.example/x.gguf"), "HTTPS"),
        (lambda d: d["models"][0].update(url="http://huggingface.co/x.gguf"), "HTTPS"),
        (lambda d: d["models"][0].update(sha256="abc"), "SHA-256"),
        (lambda d: d["models"][0].update(size=-1), "positive"),
        (lambda d: d["models"][0].update(size="big"), "int"),
        (lambda d: d["models"][0].update(filename="../x.gguf"), "plain .gguf"),
        (lambda d: d["models"][0].update(filename="x.bin"), "plain .gguf"),
        (lambda d: d["models"][0]["architecture"].update(layers=0), "architecture"),
        (lambda d: d["models"].append(dict(d["models"][0])), "duplicate"),
        (lambda d: d.update(catalog_version=999), "newer"),
        (lambda d: d.update(kind="runtimes"), "not a models catalog"),
    ],
)
def test_malformed_model_catalog_is_rejected(mutate, message):
    with pytest.raises(cat.CatalogError, match=message):
        cat.load_models(_with("models", mutate))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["runtimes"][0].update(executable="../llama-server"),
        lambda d: d["runtimes"][0].update(os="plan9"),
        lambda d: d["runtimes"][0].update(assets=[]),
        lambda d: d["runtimes"][0]["assets"][0].update(format="rar"),
        lambda d: d["runtimes"][0]["assets"][0].update(url="https://example.com/a.zip"),
    ],
)
def test_malformed_runtime_catalog_is_rejected(mutate):
    with pytest.raises(cat.CatalogError):
        cat.load_runtimes(_with("runtimes", mutate))


def test_invalid_json_is_rejected():
    with pytest.raises(cat.CatalogError, match="JSON"):
        cat.load_models("{not json")


# --- hardware -------------------------------------------------------------------------


def test_scan_hardware_is_fast_and_serialisable(tmp_path):
    profile = scan_hardware(tmp_path)
    assert profile.total_ram_bytes and profile.logical_cores
    assert profile.scan_ms < 5000
    again = HardwareProfile.from_dict(json.loads(json.dumps(profile.as_dict())))
    assert again == profile
    assert "torch" not in sys.modules  # detection never imports torch


def _card(root, n, vendor, device, cls=0x030000, vram=None):
    pci = root / f"pci{n}"
    pci.mkdir(parents=True)
    (pci / "vendor").write_text(f"{vendor:#06x}\n", encoding="ascii")
    (pci / "device").write_text(f"{device:#06x}\n", encoding="ascii")
    (pci / "class").write_text(f"{cls:#08x}\n", encoding="ascii")
    if vram is not None:
        (pci / "mem_info_vram_total").write_text(str(vram), encoding="ascii")
        (pci / "mem_info_vram_used").write_text(str(vram // 4), encoding="ascii")
    card = root / "drm" / f"card{n}"
    card.mkdir(parents=True)
    (card / "device").symlink_to(pci)


@pytest.mark.posix_only
def test_sysfs_detection_reports_confidence_without_guessing(tmp_path):
    _card(tmp_path, 0, 0x10DE, 0x1F95)  # NVIDIA without NVML: VRAM unknown
    _card(tmp_path, 1, 0x1002, 0x73BF, vram=16 * GiB)  # discrete AMD: confirmed
    _card(tmp_path, 2, 0x1002, 0x1636, vram=512 * 1024**2)  # APU carve-out
    _card(tmp_path, 3, 0x8086, 0x9A49)  # Intel iGPU
    _card(tmp_path, 4, 0x1B36, 0x0100, cls=0x020000)  # network card: ignored
    (tmp_path / "drm" / "card0-HDMI-A-1").mkdir()  # connector: ignored
    gpus = {
        g.vendor + str(g.integrated): g for g in gpu_mod.sysfs_gpus(tmp_path / "drm")
    }
    assert len(gpus) == 4
    nv = gpus["nvidiaFalse"]
    assert nv.vram_bytes is None and nv.confidence == "unknown"
    amd = gpus["amdFalse"]
    assert amd.vram_bytes == 16 * GiB and amd.confidence == "confirmed"
    assert amd.vram_free_bytes == 12 * GiB
    assert gpus["amdTrue"].vram_bytes is None and gpus["intelTrue"].integrated


def test_missing_nvml_and_sysfs_mean_no_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_mod, "_load_nvml", lambda: None)
    assert gpu_mod.nvml_gpus() == []
    assert gpu_mod.sysfs_gpus(tmp_path / "nothing") == []


# --- planning -------------------------------------------------------------------------


def test_kv_cache_formula():
    arch = {"layers": 36, "kv_heads": 8, "head_dim": 128}
    assert kv_bytes(arch, 8192, "f16") == 2 * 36 * 8 * 128 * 8192 * 2
    assert kv_bytes(arch, 8192, "q8_0") == int(2 * 36 * 8 * 128 * 8192 * 1.0625)


def test_vram_margin_and_unknown_vram():
    assert usable_vram(nvidia(4)) == 4 * GiB - int(4 * GiB * 0.10)
    margin = int(4 * GiB * 0.10)
    live = usable_vram(nvidia(4, free_gb=0.6), live=True)  # what is free right now
    assert live == int(0.6 * GiB) - margin
    assert usable_vram(nvidia(4, free_gb=0.3), live=True) == 0
    assert usable_vram(nvidia(4, confidence="unknown")) == 0
    assert usable_vram(integrated()) == 0


def test_cpu_8gb_recommends_the_light_model():
    best, fits = pick(hw(ram_gb=8, cores=4))
    assert best.model_id == "qwen3-1.7b-q8_0" and best.plan.placement == "cpu"
    assert fits["qwen3-8b-q4_k_m"].rating == "not_recommended"
    assert best.plan.est_ram_bytes < 8 * GiB - 3 * GiB


def test_cpu_16gb_eight_cores_recommends_4b_on_cpu():
    best, fits = pick(hw(ram_gb=16, cores=8))
    assert best.model_id == "qwen3-4b-q4_k_m" and best.plan.gpu_layers == 0
    assert fits["qwen3-8b-q4_k_m"].rating == "slow"
    assert any("CPU" in r for r in best.reasons)


def test_4gb_vram_16gb_ram_fits_4b_fully_on_gpu():
    best, _ = pick(hw(gpus=[nvidia(4)]))
    assert best.model_id == "qwen3-4b-q4_k_m"
    assert best.plan.placement == "gpu" and best.plan.gpu_layers == 999
    assert best.plan.context == 8192 and best.plan.kv_type == "q8_0"
    assert best.plan.est_vram_bytes <= usable_vram(nvidia(4))


@pytest.mark.parametrize("vram,ram", [(8, 32), (12, 32)])
def test_bigger_gpus_recommend_8b(vram, ram):
    best, _ = pick(hw(ram_gb=ram, gpus=[nvidia(vram)]))
    assert best.model_id == "qwen3-8b-q4_k_m" and best.plan.placement == "gpu"


def test_unknown_vram_falls_back_to_cpu_and_says_why():
    profile = hw(gpus=[nvidia(None, confidence="unknown")])
    assert available_backends(profile, ["cpu", "cuda"]) == ["cpu"]
    best, _ = pick(profile)
    assert best.plan.gpu_layers == 0
    assert any("could not be measured" in r for r in best.reasons)


def test_low_disk_marks_downloads_not_recommended_unless_installed():
    profile = hw(disk_gb=3)
    fits = {f.model_id: f for f in recommend(MODELS, profile, ["cpu"])}
    assert fits["qwen3-4b-q4_k_m"].rating == "not_recommended"
    assert "disk space" in fits["qwen3-4b-q4_k_m"].reasons[0]
    installed = recommend(
        MODELS, profile, ["cpu"], installed_ids=frozenset({"qwen3-4b-q4_k_m"})
    )
    assert {f.model_id: f for f in installed}[
        "qwen3-4b-q4_k_m"
    ].rating != "not_recommended"


def test_integrated_only_uses_cpu():
    profile = hw(gpus=[integrated()])
    assert available_backends(profile, ["cpu", "cuda"]) == ["cpu"]
    best, _ = pick(profile)
    assert best.plan.placement == "cpu"


def test_multi_gpu_uses_the_largest_discrete_gpu():
    profile = hw(
        ram_gb=32, gpus=[integrated(), nvidia(4, name="A"), nvidia(12, name="B")]
    )
    assert profile.best_gpu.name == "B"
    best, _ = pick(profile)
    assert best.model_id == "qwen3-8b-q4_k_m" and best.plan.placement == "gpu"


def test_recommendation_never_depends_on_the_gpu_name():
    names = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce GT 710", "Mystery Card", ""]
    results = {
        (
            pick(hw(gpus=[nvidia(4, name=n)]))[0].model_id,
            pick(hw(gpus=[nvidia(4, name=n)]))[0].plan.gpu_layers,
        )
        for n in names
    }
    assert len(results) == 1


def test_no_ram_measurement_means_nothing_is_recommended_as_runnable():
    profile = hw()
    profile.total_ram_bytes = None
    fits = recommend(MODELS, profile, ["cpu"])
    assert all(f.rating == "not_recommended" for f in fits)


def test_fallback_ladder_gets_strictly_safer():
    first = plan_launch(QWEN4B, hw(gpus=[nvidia(4)]), backend="cuda")
    steps = fallback_plans(QWEN4B, first)
    labels = [label for _, label in steps]
    assert labels[0].startswith("reduced the context")
    assert any("CPU only" in label for label in labels)
    assert steps[-1][0].kv_type == "f16" and steps[-1][0].context == 2048
    assert steps[-1][0].gpu_layers == 0
    contexts = [p.context for p, _ in steps]
    assert contexts == sorted(contexts, reverse=True)
