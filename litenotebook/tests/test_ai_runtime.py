"""Phase 12: the llama-server runtime manager, models and the LocalAI service.

llama-server is replaced by ``fake_llama_server.py`` (started through the
manager's ``launcher`` or as a fake ``llama-server`` script); downloads come
from a local HTTP server. Nothing here touches the user's own llama-server,
the GPU or the internet.
"""

import dataclasses
import io
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import httpx
import pytest

from ai_fixtures import (
    FileServer,
    fake_gguf,
    fake_state,
    launcher,
    make_fake_runtime,
    sha256,
)
from conftest import process_gone
from litenotebook.ai import catalog as cat
from litenotebook.ai.download import Downloader
from litenotebook.ai.hardware import GPUInfo, HardwareProfile
from litenotebook.ai.models import (
    ModelError,
    import_model,
    model_status,
    read_gguf,
    remove_model,
)
from litenotebook.ai.runtime_manager import (
    LaunchSpec,
    RuntimeManager,
    RuntimeStartError,
    classify,
    free_port,
)
from litenotebook.ai.runtimes import (
    install_runtime,
    list_installed,
    probe,
    runtime_root,
)
from litenotebook.ai.service import LocalAI, SetupError
from litenotebook.ai.store import AIStore
from litenotebook.platform_utils.system import executable_name, os_name

GiB = 1024**3
posix_only = pytest.mark.posix_only


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "fake-state.json"
    monkeypatch.setenv("FAKE_LLAMA_STATE", str(path))
    monkeypatch.setenv("FAKE_LLAMA_LOAD_S", "0.3")
    monkeypatch.delenv("FAKE_LLAMA_MODE", raising=False)
    return path


@pytest.fixture
def model_file(tmp_path):
    return fake_gguf(tmp_path / "external" / "tiny.gguf")


@pytest.fixture
def manager(settings, state_file):
    m = RuntimeManager(
        settings, launcher=launcher(), ready_timeout_s=20, stop_grace_s=3
    )
    yield m
    m.stop()


def spec(model_file, *, port=None, gpu_layers=0, context=4096):
    return LaunchSpec(
        executable="llama-server",
        model_path=str(model_file),
        alias="test/tiny",
        port=port or free_port(),
        context=context,
        gpu_layers=gpu_layers,
        threads=2,
        parallel=1,
        flash_attn="on",
        kv_type="q8_0",
        runtime_args=("--jinja",),
    )


# --- runtime manager -----------------------------------------------------------------


def test_start_waits_for_health_then_serves_and_stops_cleanly(
    manager, model_file, state_file
):
    s = spec(model_file)
    status = manager.start(s)
    assert status["state"] == "running" and status["port"] == s.port
    assert status["load_s"] >= 0.3  # waited through the 503 "loading" phase
    models = httpx.get(f"http://127.0.0.1:{s.port}/v1/models").json()
    assert models["data"][0]["id"] == "test/tiny"
    argv = fake_state(state_file)["argv"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--jinja" in argv and argv[argv.index("-ctk") + 1] == "q8_0"
    pid = status["pid"]
    assert manager.stop() in ("terminated", "exited", "killed")
    assert process_gone(pid) and manager.status()["state"] == "stopped"
    assert (manager.log_path).read_text(encoding="utf-8").startswith("$ ")


def test_cpu_plan_hides_the_gpu_from_the_process(manager, model_file, state_file):
    manager.start(spec(model_file, gpu_layers=0))
    assert fake_state(state_file)["cuda_visible_devices"] == ""


def test_gpu_plan_leaves_the_environment_alone(
    manager, model_file, state_file, monkeypatch
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    manager.start(spec(model_file, gpu_layers=999))
    assert fake_state(state_file)["cuda_visible_devices"] is None


@pytest.mark.parametrize(
    "mode,kind",
    [("oom_gpu", "out_of_memory"), ("crash", "crash"), ("port_error", "port")],
)
def test_early_exit_is_classified(manager, model_file, monkeypatch, mode, kind):
    monkeypatch.setenv("FAKE_LLAMA_MODE", mode)
    with pytest.raises(RuntimeStartError) as info:
        manager.start(spec(model_file, gpu_layers=999))
    assert info.value.kind == kind
    assert manager.status()["state"] == "failed" and manager.status()["last_error"]


def test_not_ready_in_time_is_stopped(settings, model_file, state_file, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    m = RuntimeManager(
        settings, launcher=launcher(), ready_timeout_s=1.5, stop_grace_s=2
    )
    with pytest.raises(RuntimeStartError) as info:
        m.start(spec(model_file))
    assert info.value.kind == "timeout"
    assert process_gone(fake_state(state_file)["pid"])


def test_status_answers_and_stop_cancels_while_loading(
    settings, model_file, state_file, monkeypatch
):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    m = RuntimeManager(
        settings, launcher=launcher(), ready_timeout_s=60, stop_grace_s=2
    )
    errors = []
    t = threading.Thread(
        target=lambda: errors.append(_start_error(m, spec(model_file)))
    )
    t.start()
    deadline = time.monotonic() + 10
    while m.status()["state"] != "starting" and time.monotonic() < deadline:
        time.sleep(0.05)
    started = time.monotonic()
    assert m.status()["state"] == "starting"  # not blocked by the loading start()
    assert time.monotonic() - started < 1
    m.stop()
    t.join(10)
    assert errors and errors[0] == "cancelled"
    assert process_gone(fake_state(state_file)["pid"])


def _start_error(m, s):
    try:
        m.start(s)
    except RuntimeStartError as e:
        return e.kind
    return None


def test_occupied_port_is_left_alone_and_another_port_is_used(manager, model_file):
    with socket.socket() as other:  # "someone else's" server
        other.bind(("127.0.0.1", 0))
        other.listen()
        busy = other.getsockname()[1]
        notes = []
        status = manager.start(spec(model_file, port=busy), on_progress=notes.append)
        assert status["port"] != busy
        assert any(f"Port {busy} is used by another program" in n for n in notes)
        other.settimeout(2)
        with socket.create_connection(("127.0.0.1", busy)):
            conn, _ = other.accept()  # still ours to accept: never touched
            conn.close()


def test_unrelated_processes_are_never_stopped(manager, model_file):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        manager.start(spec(model_file))
        manager.stop()
        manager.stop()  # idempotent
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()


def test_restart_with_the_same_spec_reuses_the_running_process(manager, model_file):
    s = spec(model_file)
    first = manager.start(s)["pid"]
    assert manager.start(s)["pid"] == first


def test_missing_model_file_fails_before_starting(manager, tmp_path):
    with pytest.raises(RuntimeStartError) as info:
        manager.start(spec(tmp_path / "nope.gguf"))
    assert info.value.kind == "model"


def test_classify_log_lines():
    assert classify(["CUDA error: out of memory"])[0] == "out_of_memory"
    assert classify(["error: couldn't bind HTTP server socket"])[0] == "port"
    assert classify(["llama_model_load: error loading model"])[0] == "model"
    assert classify(["something else"])[0] == "crash"


# --- models -------------------------------------------------------------------------


def test_read_gguf_metadata(model_file, tmp_path):
    info = read_gguf(model_file)
    assert info.architecture == {
        "layers": 4,
        "kv_heads": 2,
        "head_dim": 64,
        "max_context": 4096,
    }
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOTGGUF" + b"\0" * 100)
    with pytest.raises(ModelError):
        read_gguf(bad)


def test_import_in_place_is_never_deleted(settings, conn, model_file):
    store = AIStore(conn)
    record = import_model(settings, store, model_file)
    assert (
        record.owned is False and record.path == str(model_file) and not record.verified
    )
    assert model_status(record) == "ok"
    assert remove_model(settings, store, record.id) is False
    assert model_file.exists() and store.models() == []


def test_import_copy_is_owned_and_deleted_on_remove(settings, conn, model_file):
    store = AIStore(conn)
    record = import_model(settings, store, model_file, copy=True)
    copied = Path(record.path)
    assert record.owned and copied.read_bytes() == model_file.read_bytes()
    assert copied.parent == settings.data_dir / "models"
    with pytest.raises(ModelError, match="already"):
        import_model(settings, store, model_file, copy=True)
    assert remove_model(settings, store, record.id) is True
    assert not copied.exists() and model_file.exists()


def test_a_tampered_record_cannot_delete_files_outside_the_models_folder(
    settings, conn, model_file
):
    store = AIStore(conn)
    record = import_model(settings, store, model_file)
    store.save_models([dataclasses.replace(record, owned=True)])  # pretend we own it
    assert remove_model(settings, store, record.id) is False
    assert model_file.exists()


def test_import_rejects_non_gguf(settings, conn, tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("hello", encoding="utf-8")
    with pytest.raises(ModelError):
        import_model(settings, AIStore(conn), other)


# --- LocalAI service end to end (fake runtime) ----------------------------------------


def nvidia_hw(vram_gb=4.0, free_gb=None):
    vram = int(vram_gb * GiB)
    gpu = GPUInfo(
        "nvidia",
        "Any",
        vram,
        int((free_gb or vram_gb) * GiB),
        False,
        "confirmed",
        "nvml",
        "570.0",
        ["cuda", "cpu"],
    )
    return HardwareProfile(
        "linux", "Linux", "x86_64", "CPU", 8, 16, 16 * GiB, 12 * GiB, 100 * GiB, [gpu]
    )


@pytest.fixture
def service(settings, state_file, monkeypatch):
    runtime = RuntimeManager(
        settings, launcher=launcher(), ready_timeout_s=20, stop_grace_s=3
    )
    svc = LocalAI(settings, runtime)
    monkeypatch.setattr(svc, "hardware", lambda conn, rescan=False: nvidia_hw())
    yield svc
    runtime.stop()


@posix_only
def test_existing_llama_server_import_select_test(
    service, conn, tmp_path, model_file, state_file
):
    exe = make_fake_runtime(tmp_path / "bin")
    profile = service.use_existing_runtime(conn, str(exe))
    assert profile.type == "external_llama" and profile.backend == "cuda"
    record = service.import_model(conn, str(model_file))
    assert AIStore(conn).profile().model_id == record.id
    profile = AIStore(conn).profile()
    assert profile.gpu_layers == 999  # small model, 4 GiB GPU: all layers
    outcome = service.test(conn, None)
    assert outcome.ok, outcome.message
    assert outcome.reply == "ready"  # <think> removed
    assert outcome.tokens_per_s == 42.5 and outcome.ttft_s is not None
    saved = AIStore(conn).profile()
    assert saved.last_test["ok"] and "setup" not in saved.note
    assert "test" in AIStore(conn).setup().done
    assert service.runtime.running()
    endpoint = service.generation_endpoint(conn, None)
    assert endpoint.base_url == service.runtime.base_url() and endpoint.managed
    service.stop()
    assert process_gone(fake_state(state_file)["pid"])


@posix_only
def test_out_of_memory_falls_back_to_safer_settings(
    service, conn, tmp_path, model_file, monkeypatch
):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "oom_gpu")
    service.use_existing_runtime(conn, str(make_fake_runtime(tmp_path / "bin")))
    service.import_model(conn, str(model_file))
    outcome = service.test(conn, None)
    assert outcome.ok, outcome.message
    assert outcome.changes and outcome.changes[-1] in (
        "ran the model on the CPU only",
        "used the most compatible KV-cache settings with a 2,048-token context",
    )
    profile = AIStore(conn).profile()
    assert profile.gpu_layers == 0 and "did not fit in memory" in profile.note


@posix_only
def test_when_every_fallback_fails_the_user_is_told_to_choose_smaller(
    service, conn, tmp_path, model_file, monkeypatch
):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "crash")
    service.use_existing_runtime(conn, str(make_fake_runtime(tmp_path / "bin")))
    service.import_model(conn, str(model_file))
    outcome = service.test(conn, None)
    assert not outcome.ok and "stopped while starting" in outcome.message
    assert not service.runtime.running()


@posix_only
def test_port_change_on_start_is_saved_and_used(service, conn, tmp_path, model_file):
    service.use_existing_runtime(conn, str(make_fake_runtime(tmp_path / "bin")))
    service.import_model(conn, str(model_file))
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        store = AIStore(conn)
        profile = store.profile()
        profile.port = other.getsockname()[1]
        store.save_profile(profile)
        busy = profile.port
        status = service.start(conn)
    assert AIStore(conn).profile().port == status["port"] != busy
    endpoint = service.generation_endpoint(conn, None)
    assert endpoint.base_url == f"http://127.0.0.1:{status['port']}/v1"


def test_managed_runtime_must_be_inside_the_runtime_folder(service, conn, model_file):
    service.import_model(conn, str(model_file))
    store = AIStore(conn)
    profile = store.profile()
    profile.type, profile.executable = "managed_llama", "../../../../bin/sh"
    store.save_profile(profile)
    with pytest.raises(SetupError, match="missing"):
        service.launch_spec(conn)


def test_launch_refuses_a_non_gguf_model(service, conn, model_file, tmp_path):
    record = service.import_model(conn, str(model_file))
    model_file.write_bytes(b"#!/bin/sh\necho pwned\n")  # replaced after import
    store = AIStore(conn)
    profile = store.profile()
    profile.type, profile.executable = "external_llama", sys.executable
    store.save_profile(profile)
    with pytest.raises(SetupError):
        service.launch_spec(conn)
    assert record.id


def test_none_profile_has_no_endpoint_and_autostart_off_refuses(
    service, conn, model_file
):
    service.skip_generation(conn)
    assert service.generation_endpoint(conn, None) is None
    store = AIStore(conn)
    profile = store.profile()
    profile.type, profile.autostart = "external_llama", False
    store.save_profile(profile)
    with pytest.raises(SetupError, match="stopped"):
        service.ensure_running(conn)
    assert not service.runtime.running()


def test_select_unknown_model_and_remove_selected(service, conn, model_file):
    with pytest.raises(SetupError):
        service.select_model(conn, "nope")
    record = service.import_model(conn, str(model_file))
    assert service.remove_model(conn, record.id) is False
    assert AIStore(conn).profile().model_id is None and model_file.exists()


# --- downloads into the real install pipeline (local server) ------------------------


@pytest.fixture
def files():
    server = FileServer()
    yield server
    server.close()


def local_downloader():
    return Downloader(
        require_https=False,
        allowed_hosts=frozenset({"127.0.0.1"}),
        reserve_bytes=0,
        backoff_s=0.01,
    )


def test_install_model_from_the_catalog_verifies_and_selects(
    service, conn, files, monkeypatch, tmp_path
):
    body = fake_gguf(tmp_path / "src.gguf").read_bytes()
    entry = dataclasses.replace(
        cat.model("qwen3-1.7b-q8_0"),
        url=files.add("/q.gguf", body),
        sha256=sha256(body),
        size=len(body),
    )
    monkeypatch.setattr(cat, "model", lambda model_id: entry)
    service.downloader_factory = local_downloader
    record = service.install_model(conn, entry.id)
    assert record.owned and record.verified and Path(record.path).read_bytes() == body
    assert AIStore(conn).profile().model_id == entry.id
    assert service.remove_model(conn, entry.id) is True
    assert not Path(record.path).exists()


def _runtime_tarball(tmp_path):
    exe = make_fake_runtime(tmp_path / "stage" / "llama-b1")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(exe, arcname="llama-b1/llama-server")
        lib = tmp_path / "stage" / "llama-b1" / "libggml.so.0"
        lib.write_bytes(b"lib")
        tf.add(lib, arcname="llama-b1/libggml.so.0")
        link = tarfile.TarInfo("llama-b1/libggml.so")
        link.type, link.linkname = tarfile.SYMTYPE, "libggml.so.0"
        tf.addfile(link)
    return buf.getvalue()


@posix_only
def test_install_runtime_end_to_end_then_start(
    service, conn, files, tmp_path, model_file, state_file
):
    body = _runtime_tarball(tmp_path)
    base = cat.runtimes_for(os_name(), "x86_64")[0]
    entry = dataclasses.replace(
        base,
        id="llama.cpp-test-cpu",
        backend="cpu",
        executable="llama-b1/llama-server",
        assets=(
            cat.Asset(files.add("/rt.tar.gz", body), len(body), sha256(body), "tar.gz"),
        ),
    )
    stages = []
    installed = install_runtime(
        service.settings,
        entry,
        downloader=local_downloader(),
        progress=lambda stage, p: stages.append(stage),
    )
    assert installed.id == entry.id and Path(installed.executable).is_file()
    assert "Extracting…" in stages and "Checking the runtime…" in stages
    root = runtime_root(service.settings)
    assert not list(root.glob(".staging-*")) and [
        r.id for r in list_installed(service.settings)
    ] == [entry.id]
    assert not (root / entry.id / "llama-b1" / "libggml.so").is_symlink()
    assert not list((service.settings.data_dir / "tmp" / "downloads").glob("*"))
    # use it: managed profile, started without the test launcher
    store = AIStore(conn)
    profile = store.profile()
    profile.type, profile.backend = "managed_llama", "cpu"
    profile.executable = Path(installed.executable).relative_to(root).as_posix()
    store.save_profile(profile)
    service.import_model(conn, str(model_file))
    service.runtime.launcher = None
    status = service.start(conn)
    assert status["state"] == "running"
    service.stop()


def test_install_runtime_with_a_bad_checksum_installs_nothing(service, files, tmp_path):
    body = b"definitely not a tarball"
    base = cat.runtimes_for(os_name(), "x86_64")[0]
    entry = dataclasses.replace(
        base,
        id="llama.cpp-bad",
        assets=(cat.Asset(files.add("/bad.tgz", body), len(body), "0" * 64, "tar.gz"),),
    )
    with pytest.raises(Exception, match="checksum"):
        install_runtime(service.settings, entry, downloader=local_downloader())
    assert list_installed(service.settings) == []


# --- existing llama.cpp and simulated Windows ----------------------------------------


@posix_only
def test_probe_runs_version_without_a_shell(tmp_path):
    exe = make_fake_runtime(tmp_path / "bin")
    result = probe(exe)
    assert result.ok and result.build == 11146 and result.backend == "cuda"
    assert result.devices == ["CUDA0: Fake GPU (4096 MiB, 3000 MiB free)"]


def test_probe_refuses_other_programs(tmp_path):
    other = tmp_path / "bash"
    other.write_text("#!/bin/sh\n", encoding="utf-8")
    assert not probe(other).ok and "llama-server" in probe(other).error
    assert not probe(tmp_path / "missing" / "llama-server").ok
    semicolon = tmp_path / "llama-server; rm -rf ~"
    semicolon.write_text("", encoding="utf-8")
    assert not probe(semicolon).ok  # names are never passed to a shell


def test_simulated_windows_names_and_catalog():
    assert executable_name("llama-server", system="win32") == "llama-server.exe"
    assert executable_name("llama-server.exe", system="win32") == "llama-server.exe"
    assert executable_name("llama-server", system="linux") == "llama-server"
    win = cat.runtimes_for("windows", "x86_64")
    assert win and all(r.executable == "llama-server.exe" for r in win)
    assert all(a.format == "zip" for r in win for a in r.assets)
    cuda = next(r for r in win if r.backend == "cuda")
    assert len(cuda.assets) == 2  # the build and the CUDA runtime DLLs
    s = LaunchSpec(
        r"C:\Users\A B\AppData\Local\LiteNotebook\runtime\llama-server.exe",
        r"D:\Models\qwen 3.gguf",
        "a",
        8081,
        4096,
        0,
        4,
        1,
        "on",
        "q8_0",
    )
    argv = s.argv()
    assert argv[0].endswith("llama-server.exe") and argv[2] == r"D:\Models\qwen 3.gguf"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


@posix_only
def test_autostart_replans_for_live_memory_and_falls_back(
    service, conn, tmp_path, model_file, monkeypatch
):
    """Regression (real Linux run): the stored plan used the GPU's total VRAM;
    another program held the GPU, so Chat's autostart ran out of memory."""
    service.use_existing_runtime(conn, str(make_fake_runtime(tmp_path / "bin")))
    service.import_model(conn, str(model_file))
    assert AIStore(conn).profile().gpu_layers == 999  # stored: all on the GPU
    monkeypatch.setenv("FAKE_LLAMA_MODE", "oom_gpu")
    service.ensure_running(conn)  # what Chat calls
    assert service.runtime.running()
    profile = AIStore(conn).profile()
    assert profile.gpu_layers == 0 and "did not fit in memory" in profile.note


@posix_only
def test_autostart_uses_live_free_vram(
    service, conn, tmp_path, model_file, monkeypatch, state_file
):
    service.use_existing_runtime(conn, str(make_fake_runtime(tmp_path / "bin")))
    service.import_model(conn, str(model_file))
    busy = nvidia_hw(vram_gb=4, free_gb=0.2)  # someone else holds the GPU
    monkeypatch.setattr(service, "hardware", lambda conn, rescan=False: busy)
    service.ensure_running(conn)
    argv = fake_state(state_file)["argv"]
    assert argv[argv.index("-ngl") + 1] == "0"  # planned for CPU, first try
    assert fake_state(state_file)["cuda_visible_devices"] == ""


@posix_only
def test_llama_server_never_outlives_a_killed_litenotebook(
    tmp_path, model_file, state_file
):
    """SIGKILL (or a closed terminal) skips all shutdown code; the supervise
    guard must still stop llama-server once its pipe from the parent closes."""
    import os
    import signal

    script = tmp_path / "parent.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from ai_fixtures import launcher\n"
        "from litenotebook.ai.runtime_manager import (\n"
        "    LaunchSpec, RuntimeManager, free_port)\n"
        "from litenotebook.config import load_settings\n"
        f"data = {str(tmp_path / 'data')!r}\n"
        "s = load_settings({'LITENOTEBOOK_DATA_DIR': data})\n"
        "m = RuntimeManager(s, launcher=launcher(), ready_timeout_s=20)\n"
        f"st = m.start(LaunchSpec('llama-server', {str(model_file)!r}, 'a',"
        " free_port(),"
        " 2048, 0, 1, 1, 'on', 'q8_0'))\n"
        "print(st['pid'], flush=True)\n"
        "import time; time.sleep(60)\n",
        encoding="utf-8",
    )
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE)
    try:
        llama_pid = int(parent.stdout.readline())
        assert llama_pid == fake_state(state_file)["pid"]  # the real child is reported
        assert not process_gone(llama_pid, timeout=0.5)
        os.kill(parent.pid, signal.SIGKILL)  # no shutdown code runs
        parent.wait(10)
        assert process_gone(llama_pid, timeout=15)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
