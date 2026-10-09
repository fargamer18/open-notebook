"""Phase 15: desktop packaging logic, tested in source mode with simulated
frozen builds (the built packages are tested by tools/packaged_smoke.py)."""

import io
import json
import socket
import sqlite3
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import litenotebook
from conftest import FakeEmbedder
from litenotebook import desktop, packaging
from litenotebook.applog import RedactSecrets, configure_file_logging
from litenotebook.config import load_settings
from litenotebook.db import available_migrations, open_db, schema_version
from litenotebook.platform_utils.filesystem import InstanceLock
from litenotebook.platform_utils.system import app_dirs, executable_name
from litenotebook.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:5056"
ORIGIN = {"Origin": BASE}


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """Pretend to be a packaged app living in tmp_path/app."""
    app = tmp_path / "app"
    app.mkdir()
    exe = app / executable_name("litenotebook")
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.delenv(packaging.WORKER_ENV, raising=False)
    return app


def test_version_has_one_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{litenotebook.__version__}"' in pyproject
    assert litenotebook.__version__.startswith("0.")  # pre-release until v1 is approved


def test_source_mode_worker_command(settings):
    assert not packaging.is_frozen()
    assert packaging.embed_worker_command(settings)[1:] == [
        "-m",
        "litenotebook.embed.worker",
    ]
    assert packaging.runtime_status(settings)["packaged"] is False


def test_frozen_worker_discovery_order(frozen, settings, tmp_path, monkeypatch):
    with pytest.raises(packaging.EmbeddingRuntimeMissingError) as e:
        packaging.embed_worker_command(settings)
    assert "Settings → Local AI" in str(e.value) and "/" + "home" not in str(e.value)
    assert packaging.runtime_status(settings)["installed"] is False
    # installed in the data folder, for this version only
    installed = packaging.installed_runtime_dir(settings)
    installed.mkdir(parents=True)
    worker = installed / executable_name(packaging.WORKER_BASE)
    worker.write_bytes(b"")
    assert packaging.find_embed_worker(settings) is None  # no marker yet
    (installed / packaging.RUNTIME_MARKER).write_text(
        json.dumps({"version": "0.0.1"}), encoding="utf-8"
    )
    assert packaging.find_embed_worker(settings) is None  # another version
    (installed / packaging.RUNTIME_MARKER).write_text(
        json.dumps({"version": litenotebook.__version__}), encoding="utf-8"
    )
    assert packaging.embed_worker_command(settings) == [str(worker)]
    # beside the app wins over the data folder
    beside = frozen / packaging.RUNTIME_DIRNAME
    beside.mkdir()
    (beside / executable_name(packaging.WORKER_BASE)).write_bytes(b"")
    assert packaging.find_embed_worker(settings)[1] == "beside the app"
    # an explicit override wins over both
    explicit = tmp_path / "custom worker"
    explicit.write_bytes(b"")
    monkeypatch.setenv(packaging.WORKER_ENV, str(explicit))
    assert packaging.find_embed_worker(settings) == (
        explicit,
        "LITENOTEBOOK_EMBED_WORKER",
    )


def test_frozen_internal_helpers_only(frozen):
    from litenotebook.platform_utils.system import python_module_command

    assert python_module_command("litenotebook.platform_utils.supervise") == [
        sys.executable,
        "--internal",
        "litenotebook.platform_utils.supervise",
    ]
    with pytest.raises(RuntimeError):
        python_module_command("litenotebook.cli")
    assert desktop.entry(["--internal", "os"]) == 2  # not an allowed module


def test_frozen_app_is_never_development_mode(frozen, monkeypatch):
    from litenotebook import config

    monkeypatch.setattr(config, "PROJECT_ROOT", ROOT)  # even inside a checkout
    assert config.is_source_checkout() is False
    s = load_settings({"HOME": "/home/x", "XDG_DATA_HOME": "/xdg/data"})
    assert s.data_dir_origin == config.ORIGIN_PLATFORM


def test_platform_log_folders():
    win = app_dirs(
        {"LOCALAPPDATA": r"C:\Users\Normal User\AppData\Local"}, system="windows"
    )
    assert str(win.log) == r"C:\Users\Normal User\AppData\Local\LiteNotebook\Logs"
    linux = app_dirs({"XDG_STATE_HOME": "/home/josé/.local/state"}, system="linux")
    assert str(linux.log) == "/home/josé/.local/state/LiteNotebook/logs"
    default = app_dirs({}, system="linux", home=Path("/home/u"))
    assert str(default.log) == "/home/u/.local/state/LiteNotebook/logs"
    s = load_settings(
        {"LITENOTEBOOK_DATA_DIR": "/tmp/x", "LITENOTEBOOK_LOG_DIR": "/tmp/l"}
    )
    assert s.log_path == Path("/tmp/l")
    assert load_settings({"LITENOTEBOOK_DATA_DIR": "/tmp/x"}).log_path == Path(
        "/tmp/x/logs"
    )


# --- installing the embedding runtime from its archive ----------------------------


def runtime_archive(tmp_path: Path, *, extra: dict[str, bytes] | None = None) -> Path:
    path = (
        tmp_path / f"LiteNotebook-embedding-runtime-{litenotebook.__version__}.tar.gz"
    )
    with tarfile.open(path, "w:gz") as tf:
        files = {
            f"embedding-runtime/{executable_name(packaging.WORKER_BASE)}": b"#!x",
            "embedding-runtime/_internal/lib.so": b"x",
            **(extra or {}),
        }
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return path


def expected_for(path: Path) -> dict:
    import hashlib

    return {
        "file": path.name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_install_runtime_archive(frozen, settings, tmp_path):
    archive = runtime_archive(tmp_path)
    target = packaging.install_runtime_archive(
        settings, archive, expected=expected_for(archive)
    )
    assert target == packaging.installed_runtime_dir(settings)
    assert packaging.find_embed_worker(settings)[1] == "installed in the data folder"
    assert not list(target.parent.glob(".staging-*"))
    # repair: installing again replaces it cleanly
    packaging.install_runtime_archive(settings, archive, expected=expected_for(archive))
    assert packaging.find_embed_worker(settings) is not None


@pytest.mark.parametrize("problem", ["name", "size", "sha", "missing", "unlisted"])
def test_runtime_archive_must_match_the_manifest(frozen, settings, tmp_path, problem):
    archive = runtime_archive(tmp_path)
    expected = expected_for(archive)
    if problem == "name":
        expected["file"] = "other.tar.gz"
    elif problem == "size":
        expected["size"] += 1
    elif problem == "sha":
        expected["sha256"] = "0" * 64
    elif problem == "missing":
        archive = tmp_path / "nope.tar.gz"
    else:
        expected = {}
    with pytest.raises(packaging.RuntimeInstallError):
        packaging.install_runtime_archive(settings, archive, expected=expected)
    assert packaging.find_embed_worker(settings) is None


def test_hostile_runtime_archive_installs_nothing(frozen, settings, tmp_path):
    from litenotebook.ai.extract import UnsafeArchiveError

    archive = runtime_archive(tmp_path, extra={"../escape.txt": b"x"})
    with pytest.raises(UnsafeArchiveError):
        packaging.install_runtime_archive(
            settings, archive, expected=expected_for(archive)
        )
    assert packaging.find_embed_worker(settings) is None
    assert not (tmp_path / "escape.txt").exists()


# --- launcher ---------------------------------------------------------------------


def test_choose_port_never_disturbs_a_listener():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        chosen = desktop.choose_port("127.0.0.1", taken)
        assert chosen != taken and desktop.port_is_free("127.0.0.1", chosen)
        busy.getsockname()  # still ours, still listening
    free = desktop.choose_port("127.0.0.1", chosen)
    assert free == chosen


def test_second_launch_opens_the_running_instance(settings, monkeypatch):
    settings.data_dir.mkdir(parents=True)
    opened = []
    monkeypatch.setattr(desktop, "open_browser", opened.append)
    monkeypatch.setattr(desktop, "is_litenotebook", lambda url, timeout=2.0: True)
    holder = InstanceLock(settings.data_dir)
    assert holder.acquire("server", info={"url": "http://127.0.0.1:5999"})
    try:
        done = []
        t = threading.Thread(
            target=lambda: done.append(desktop.run([], settings=settings))
        )
        t.start()
        t.join(10)
    finally:
        holder.release()
    assert done == [0] and opened == ["http://127.0.0.1:5999"]


def test_busy_without_a_page_reports_an_error(settings, monkeypatch, tmp_path):
    settings.data_dir.mkdir(parents=True)
    reported = []
    monkeypatch.setattr(desktop, "show_error_dialog", lambda *a: False)
    monkeypatch.setattr(desktop, "open_browser", reported.append)
    holder = InstanceLock(settings.data_dir)
    assert holder.acquire("command")  # a CLI command: no URL
    try:
        result = []
        t = threading.Thread(
            target=lambda: result.append(desktop.run([], settings=settings))
        )
        t.start()
        t.join(10)
    finally:
        holder.release()
    assert result == [1]
    page = settings.log_path / "startup-error.html"
    assert reported and reported[0].startswith("file:") and page.is_file()
    assert "already running" in page.read_text(encoding="utf-8")
    log = (settings.log_path / "litenotebook.log").read_text(encoding="utf-8")
    assert "Start-up failed" in log


def test_desktop_run_serves_opens_and_quits(settings, monkeypatch):
    opened = []
    monkeypatch.setattr(desktop, "open_browser", opened.append)
    from litenotebook.services import embedding

    monkeypatch.setattr(
        embedding, "worker_embedder_factory", lambda s=None: FakeEmbedder
    )
    result = []
    t = threading.Thread(
        target=lambda: result.append(desktop.run(["--port", "5987"], settings=settings))
    )
    t.start()
    deadline = time.monotonic() + 30
    while not opened and time.monotonic() < deadline:
        time.sleep(0.05)
    assert opened and opened[0].startswith("http://127.0.0.1:")
    url = opened[0]
    import urllib.request

    assert (
        json.loads(urllib.request.urlopen(url + "/livez").read())["app"]
        == "LiteNotebook"
    )
    holder = InstanceLock(settings.data_dir).holder()
    assert holder["url"] == url and holder["launcher"] == "desktop"
    request = urllib.request.Request(
        url + "/app/quit", data=b"", method="POST", headers={"Origin": url}
    )
    assert urllib.request.urlopen(request).status == 200
    t.join(30)
    assert result == [0]
    log = (settings.log_path / "litenotebook.log").read_text(encoding="utf-8")
    assert "Ready at" in log and "stopped" in log


def test_entry_dispatch(capsys, monkeypatch):
    assert desktop.entry(["--version"]) == 0
    assert litenotebook.__version__ in capsys.readouterr().out
    called = []
    monkeypatch.setattr(desktop, "run", lambda argv: called.append(("run", argv)) or 0)
    import litenotebook.cli as cli

    monkeypatch.setattr(cli, "main", lambda argv: called.append(("cli", argv)) or 0)
    desktop.entry([])
    desktop.entry(["--no-browser"])
    desktop.entry(["notebook", "list"])
    assert called == [
        ("run", []),
        ("run", ["--no-browser"]),
        ("cli", ["notebook", "list"]),
    ]


@pytest.mark.posix_only
def test_linux_desktop_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    exe = tmp_path / "Normal User" / "litenotebook"
    target = desktop.install_desktop_entry(exe, tmp_path / "icon.png")
    text = target.read_text(encoding="utf-8")
    assert "Name=LiteNotebook" in text and "Terminal=false" in text
    assert f'Exec="{exe}"' in text and "Categories=Office;Education;Science;" in text
    assert desktop.uninstall_desktop_entry() and not target.exists()


# --- web additions ----------------------------------------------------------------


@pytest.fixture
def client(settings):
    app = create_app(settings, env={}, embedder_factory=FakeEmbedder)
    with TestClient(app, base_url=BASE) as c:
        yield c


def test_livez_and_quit_without_launcher(client):
    r = client.get("/livez")
    assert r.json() == {"app": "LiteNotebook", "version": litenotebook.__version__}
    r = client.post("/app/quit", headers=ORIGIN)
    assert r.status_code == 409 and "started from a terminal" in r.text


def test_quit_calls_the_launcher(client):
    called = []
    client.app.state.shutdown = lambda: called.append(True)
    r = client.post("/app/quit", headers=ORIGIN)
    assert r.status_code == 200 and "LiteNotebook is stopping" in r.text
    assert called == [True]
    evil = client.post("/app/quit", headers={"Origin": "http://evil.example"})
    assert evil.status_code == 403 and called == [True]  # cross-site: refused


def test_diagnostics_shows_packaging_and_opens_logs(client, settings, monkeypatch):
    page = client.get("/settings/diagnostics").text
    assert "LiteNotebook " + litenotebook.__version__ in page and "(source)" in page
    assert "Open logs folder" in page and "Quit LiteNotebook" not in page
    opened = []
    from litenotebook.platform_utils import system

    monkeypatch.setattr(system, "open_folder", lambda p: opened.append(p) or True)
    r = client.post(
        "/settings/diagnostics/open-logs", headers=ORIGIN, follow_redirects=False
    )
    assert r.status_code == 303 and opened == [settings.log_path]


def test_settings_offers_runtime_install_only_when_packaged(client, frozen):
    page = client.get("/settings/ai").text
    assert "Search runtime: not installed" in page


def test_favicon_and_icon_are_original_svg(client):
    r = client.get("/static/icon.svg")
    assert (
        r.status_code == 200 and r.text.startswith("<svg") and "LiteNotebook" in r.text
    )
    assert 'rel="icon" href="/static/icon.svg"' in client.get("/").text


# --- logging and migrations -------------------------------------------------------


def test_logs_redact_secrets_and_skip_access_logs(settings):
    import logging

    path = configure_file_logging(settings)
    log = logging.getLogger("litenotebook.test")
    log.warning(
        "calling with Authorization: Bearer abcdefghijklmnop and sk-SECRET123456"
    )
    log.warning("api_key=hunter2hunter2 token: zzzzzzzzzz")
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = path.read_text(encoding="utf-8")
    assert "abcdefghijklmnop" not in text and "sk-SECRET123456" not in text
    assert "hunter2hunter2" not in text and "zzzzzzzzzz" not in text
    assert text.count("[redacted]") == 4
    assert logging.getLogger("uvicorn.access").disabled
    assert isinstance(RedactSecrets(), logging.Filter)


def test_upgrade_keeps_data_and_a_safety_copy(tmp_path):
    db_path = tmp_path / "data" / "litenotebook.sqlite3"
    db_path.parent.mkdir()
    conn = sqlite3.connect(db_path)
    for version, path in available_migrations()[:8]:  # an old schema-8 database
        conn.executescript(
            f"BEGIN;\n{path.read_text(encoding='utf-8')}\n"
            f"PRAGMA user_version = {version};"
        )
        conn.commit()
    conn.execute(
        "INSERT INTO notebooks (name, embed_model, embed_dim) VALUES ('Kept', 'm', 384)"
    )
    conn.commit()
    conn.close()
    upgraded = open_db(db_path)
    try:
        assert schema_version(upgraded) == len(available_migrations())
        assert upgraded.execute("SELECT name FROM notebooks").fetchone()[0] == "Kept"
    finally:
        upgraded.close()
    copies = list(
        (tmp_path / "data" / "backups").glob("before-schema-*-from-8-*.sqlite3")
    )
    assert len(copies) == 1
    old = sqlite3.connect(copies[0])
    try:
        assert old.execute("PRAGMA user_version").fetchone()[0] == 8
        assert old.execute("SELECT name FROM notebooks").fetchone()[0] == "Kept"
    finally:
        old.close()
    open_db(db_path).close()  # already current: no second copy
    assert len(list((tmp_path / "data" / "backups").glob("*.sqlite3"))) == 1


def test_build_manifest_lists_no_bundled_optional_components():
    text = (ROOT / "tools" / "build.py").read_text(encoding="utf-8")
    assert '"bundled_optional_components": []' in text
    spec = (ROOT / "packaging" / "pyinstaller" / "litenotebook.spec").read_text("utf-8")
    for heavy in ("torch", "sentence_transformers", "transformers", "sherpa_onnx"):
        assert f'"{heavy}"' in spec  # excluded from the core app
