"""Phase 11: portability of the core application.

Three kinds of test live here, deliberately labelled:

* portable-code tests — run on every OS and exercise the real behaviour;
* simulations — ``ntpath``/``PureWindowsPath`` and a ``system="win32"``
  parameter model Windows rules on this machine. They check our *logic*
  against Windows conventions; they do NOT prove the app works on Windows
  (see docs/windows-test-plan.md for real verification);
* ``posix_only`` — genuinely POSIX behaviour, skipped on Windows.
"""

import ast
import ntpath
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import process_gone, symlink_or_skip
from litenotebook.archive.format import ArchiveError
from litenotebook.config import (
    ORIGIN_CUSTOM,
    ORIGIN_DEVELOPMENT,
    ORIGIN_PLATFORM,
    PROJECT_ROOT,
    ConfigError,
    load_settings,
)
from litenotebook.platform_utils.filesystem import (
    DataDirInUseError,
    InstanceLock,
    in_use,
    is_within,
    remove_tree,
    replace_file,
)
from litenotebook.platform_utils.paths import archive_name_problem, download_name
from litenotebook.platform_utils.processes import creation_flags, spawn, stop
from litenotebook.platform_utils.system import (
    app_dirs,
    executable_name,
    find_executable,
    os_name,
    python_module_command,
    system_info,
)

SRC = Path(__file__).parents[1] / "src" / "litenotebook"
BASE = "http://127.0.0.1:5056"


# --- application directories (simulated per OS) -------------------------------------


def test_linux_xdg_defaults_and_overrides():
    home = PurePosixPath("/home/ana")
    d = app_dirs({}, system="linux", home=home)
    assert d.data == home / ".local/share/LiteNotebook"
    assert d.config == home / ".config/LiteNotebook"
    assert d.cache == home / ".cache/LiteNotebook"
    env = {
        "XDG_DATA_HOME": "/data/x",
        "XDG_CONFIG_HOME": "/cfg",
        "XDG_CACHE_HOME": "/c",
    }
    d = app_dirs(env, system="linux", home=home)
    assert (d.data, d.config, d.cache) == (
        PurePosixPath("/data/x/LiteNotebook"),
        PurePosixPath("/cfg/LiteNotebook"),
        PurePosixPath("/c/LiteNotebook"),
    )
    relative = app_dirs({"XDG_DATA_HOME": "relative/dir"}, system="linux", home=home)
    assert relative.data == home / ".local/share/LiteNotebook"  # XDG: relative ignored


@pytest.mark.parametrize("user", ["Ana Maria", "Zoë Ångström", "李雷"])
def test_windows_localappdata_defaults(user):  # simulation
    local = f"C:\\Users\\{user}\\AppData\\Local"
    d = app_dirs(
        {"LOCALAPPDATA": local},
        system="win32",
        home=PureWindowsPath(f"C:\\Users\\{user}"),
    )
    assert isinstance(d.data, PureWindowsPath)
    assert str(d.data) == f"C:\\Users\\{user}\\AppData\\Local\\LiteNotebook"
    assert d.config == d.data and str(d.cache).endswith("LiteNotebook\\Cache")
    fallback = app_dirs({}, system="win32", home=PureWindowsPath(f"C:\\Users\\{user}"))
    assert str(fallback.data) == f"C:\\Users\\{user}\\AppData\\Local\\LiteNotebook"


def test_macos_defaults():  # simulation
    d = app_dirs({}, system="darwin", home=PurePosixPath("/Users/ana"))
    assert str(d.data) == "/Users/ana/Library/Application Support/LiteNotebook"
    assert str(d.cache) == "/Users/ana/Library/Caches/LiteNotebook"


def test_data_dir_resolution_modes(tmp_path, monkeypatch):
    dev = load_settings({})  # this repository is a source checkout
    assert (
        dev.data_dir == PROJECT_ROOT / "data"
        and dev.data_dir_origin == ORIGIN_DEVELOPMENT
    )
    assert dev.config_path == dev.data_dir / "config"
    user = load_settings(
        {
            "LITENOTEBOOK_MODE": "user",
            "XDG_DATA_HOME": str(tmp_path / "x"),
            "LOCALAPPDATA": str(tmp_path / "w"),
        }
    )
    assert user.data_dir_origin == ORIGIN_PLATFORM
    expected = (
        tmp_path / "w" if os_name() == "windows" else tmp_path / "x"
    ) / "LiteNotebook"
    assert user.data_dir == expected
    weird = tmp_path / "My Notes — Ünïcode" / ("long-" + "x" * 80)
    custom = load_settings({"LITENOTEBOOK_DATA_DIR": str(weird)})
    assert custom.data_dir == weird and custom.data_dir_origin == ORIGIN_CUSTOM
    monkeypatch.setenv("LN_TEST_ROOT", str(tmp_path))
    var = "%LN_TEST_ROOT%" if os_name() == "windows" else "$LN_TEST_ROOT"
    assert (
        load_settings({"LITENOTEBOOK_DATA_DIR": f"{var}/d"}).data_dir == tmp_path / "d"
    )
    over = load_settings(
        {
            "LITENOTEBOOK_DATA_DIR": str(tmp_path / "d"),
            "LITENOTEBOOK_CONFIG_DIR": str(tmp_path / "cfg"),
            "LITENOTEBOOK_CACHE_DIR": str(tmp_path / "cache"),
        }
    )
    assert (over.config_path, over.cache_path) == (tmp_path / "cfg", tmp_path / "cache")
    with pytest.raises(ConfigError, match="MODE"):
        load_settings({"LITENOTEBOOK_MODE": "production"})
    assert not any(tmp_path.iterdir()) or not (
        weird.exists() or (tmp_path / "x").exists()
    )


def test_platform_mode_used_outside_a_checkout(tmp_path, monkeypatch):
    import litenotebook.config as config

    monkeypatch.setattr(config, "is_source_checkout", lambda root=None: False)
    s = config.load_settings(
        {"XDG_DATA_HOME": str(tmp_path), "LOCALAPPDATA": str(tmp_path)}
    )
    assert (
        s.data_dir_origin == ORIGIN_PLATFORM and s.data_dir == tmp_path / "LiteNotebook"
    )


# --- executables, commands, browser -------------------------------------------------


def test_executable_names_and_discovery(tmp_path):
    assert executable_name("llama-server", system="win32") == "llama-server.exe"
    assert executable_name("llama-server.EXE", system="win32") == "llama-server.EXE"
    assert executable_name("llama-server", system="linux") == "llama-server"
    exe = tmp_path / executable_name("fake-tool")
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    assert find_executable("fake-tool", (tmp_path,)) == exe
    assert find_executable("definitely-not-installed-tool-xyz") is None


def test_worker_command_and_frozen_builds(monkeypatch):
    assert python_module_command("litenotebook.embed.worker") == [
        sys.executable,
        "-m",
        "litenotebook.embed.worker",
    ]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="Packaged builds are not supported yet"):
        python_module_command("litenotebook.embed.worker")


def test_creation_flags():  # simulation for Windows
    assert creation_flags(system="linux") == 0
    assert creation_flags(system="win32") == 0x08000000 | 0x00000200


def test_open_browser_uses_webbrowser(monkeypatch):
    import webbrowser

    from litenotebook.platform_utils.system import open_browser

    opened = []
    monkeypatch.setattr(
        webbrowser, "open", lambda url, new=0: opened.append(url) or True
    )
    assert open_browser("http://127.0.0.1:5056") and opened == ["http://127.0.0.1:5056"]


# --- owned processes ----------------------------------------------------------------


def py(code):
    return [sys.executable, "-c", textwrap.dedent(code)]


def test_spawn_rejects_shell_strings():
    with pytest.raises(ValueError):
        spawn("echo hi")  # type: ignore[arg-type]


def test_stop_graceful_via_stdin_eof():
    owned = spawn(py("import sys\nsys.stdin.buffer.read()\n"))
    assert stop(owned, grace_s=5) == "exited" and owned.proc.returncode == 0
    assert process_gone(owned.pid, 5)


def test_stop_terminates_a_child_that_ignores_eof():
    owned = spawn(py("import time\nwhile True: time.sleep(0.1)\n"))
    assert stop(owned, grace_s=0.5) == "terminated"
    assert process_gone(owned.pid, 5)


@pytest.mark.posix_only
def test_stop_kills_a_child_that_ignores_sigterm():
    owned = spawn(
        py("""
        import signal, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("ready", flush=True)
        while True: time.sleep(0.1)
    """)
    )
    assert owned.stdout.readline().strip() == b"ready"
    assert stop(owned, grace_s=0.5) == "killed"
    assert process_gone(owned.pid, 5)


@pytest.mark.posix_only
def test_tree_stop_ends_descendants_and_only_ours():
    bystander = subprocess.Popen(py("import time\ntime.sleep(30)\n"))
    owned = spawn(
        py("""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        while True: time.sleep(0.1)
    """),
        tree=True,
    )
    grandchild = int(owned.stdout.readline())
    assert stop(owned, grace_s=0.5) in ("terminated", "killed")
    assert process_gone(owned.pid, 5) and process_gone(grandchild, 5)
    assert bystander.poll() is None  # an unrelated process is never touched
    bystander.kill()
    bystander.wait()


def test_worker_protocol_tolerates_crlf():
    from litenotebook.embed import protocol as p

    assert p.loads(b'{"v": 1, "id": 3, "op": "ping"}\r\n')["id"] == 3
    assert p.dumps(p.request(1, "ping")).endswith(b"}\n") and b"\r" not in p.dumps(
        p.request(1, "ping")
    )


def test_worker_streams_are_binary_and_no_fork():
    worker = (SRC / "embed/worker.py").read_text(encoding="utf-8")
    assert "set_binary_mode(proto_fd)" in worker and "set_binary_mode(0)" in worker
    assert 'if __name__ == "__main__":' in worker
    client = (SRC / "embed/worker_client.py").read_text(encoding="utf-8")
    assert (
        "spawn(self.command" in client and "stop(owned or OwnedProcess(proc)" in client
    )


# --- locking ------------------------------------------------------------------------


def test_instance_lock_records_holder_and_blocks_second_holder(tmp_path):
    first = InstanceLock(tmp_path)
    assert first.acquire("server")
    holder = InstanceLock(tmp_path).holder()
    assert holder["pid"] == os.getpid() and holder["purpose"] == "server"
    with pytest.raises(DataDirInUseError, match=f"process {os.getpid()}"):
        InstanceLock(tmp_path).acquire_or_raise("command: ingest")
    assert in_use(tmp_path)
    first.release()
    assert not in_use(tmp_path)


def test_lock_recovers_after_a_crashed_holder(tmp_path):
    holder = subprocess.Popen(
        py(f"""
        import sys, time
        sys.path.insert(0, {str(SRC.parent)!r})
        from pathlib import Path
        from litenotebook.platform_utils.filesystem import InstanceLock
        lock = InstanceLock(Path({str(tmp_path)!r}))
        assert lock.acquire("server")
        print("held", flush=True)
        time.sleep(60)
    """),
        stdout=subprocess.PIPE,
    )
    assert holder.stdout.readline().strip() == b"held"
    assert in_use(tmp_path)
    holder.kill()  # a hard crash: no release() ever runs
    holder.wait()
    assert (tmp_path / InstanceLock.FILENAME).exists()  # the stale file is left …
    lock = InstanceLock(tmp_path)
    assert lock.acquire()  # … and never blocks: the OS dropped the lock
    lock.release()


def test_server_refuses_a_second_instance(tmp_path):
    from litenotebook.web.app import create_app

    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    app = create_app(
        s,
        env={},
        llm_transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": []})
        ),
    )
    with TestClient(app, base_url=BASE):
        second = create_app(
            s,
            env={},
            llm_transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"data": []})
            ),
        )
        with pytest.raises(DataDirInUseError, match="already in use"):
            with TestClient(second, base_url=BASE):
                pass
    with TestClient(second, base_url=BASE) as c:  # free again after shutdown
        assert c.get("/").status_code == 200


def test_cli_mutations_need_the_lock_but_reads_do_not(tmp_path, capsys):
    from litenotebook.cli import main

    data = tmp_path / "data"
    assert main(["--data-dir", str(data), "notebook", "create", "Alpha"]) == 0
    lock = InstanceLock(data)
    assert lock.acquire("server")
    try:
        assert main(["--data-dir", str(data), "notebook", "create", "Beta"]) == 1
        assert "already in use" in capsys.readouterr().err
        assert main(["--data-dir", str(data), "notebook", "list"]) == 0
        assert main(["--data-dir", str(data), "diagnostics", "platform"]) == 0
        out = capsys.readouterr().out
        assert "Alpha" in out and "Beta" not in out and "data_in_use: True" in out
    finally:
        lock.release()


# --- paths and names (Windows rules applied on every OS) ----------------------------


HOSTILE = [
    "..\\foo",
    "../foo",
    "C:\\foo",
    "C:foo",
    "\\\\server\\share\\x",
    "//server/share/x",
    "\\\\?\\C:\\foo",
    "file.txt::$DATA",
    "trailing.",
    "trailing ",
    "NUL",
    "nul.txt",
    "CON",
    "com1.pdf",
    "LPT9",
    "a/../b",
    "a\\..\\b",
    "mixed/sep\\x",
    "x" * 300,
    "nul\x00byte",
    "tab\tname",
]


@pytest.mark.parametrize("name", HOSTILE)
def test_hostile_names_never_pass_archive_validation(name):
    assert archive_name_problem(name) is not None, name


@pytest.mark.parametrize("name", HOSTILE)
def test_hostile_names_are_display_metadata_only(
    name, conn, settings, notebook, tmp_path
):
    from conftest import make_pdf
    from litenotebook.ingest.pipeline import display_filename, register_source

    shown = display_filename(name + ".pdf")
    assert "\x00" not in shown and "/" not in shown and "\\" not in shown
    src = register_source(
        conn,
        settings,
        notebook,
        make_pdf(tmp_path / f"p{abs(hash(name))}.pdf", [name[:40]]),
        filename=name + ".pdf",
    )
    stored = settings.data_dir / src.stored_path
    assert stored.name == f"{src.id}.pdf" and stored.is_file()
    assert src.stored_path == f"notebooks/{notebook.id}/sources/{src.id}.pdf"
    assert is_within(stored, settings.sources_dir(notebook.id))


def test_containment_is_by_components_not_string_prefix(tmp_path):
    root = tmp_path / "data"
    (root / "a").mkdir(parents=True)
    (tmp_path / "data2").mkdir()
    assert is_within(root / "a" / "x.pdf", root)
    assert not is_within(
        tmp_path / "data2" / "x.pdf", root
    )  # "data2".startswith("data")
    assert not is_within(root / ".." / "data2", root)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_or_skip(root / "link", outside, directory=True)
    assert not is_within(root / "link" / "x", root)


def test_ntpath_rules_we_rely_on():  # simulation of Windows path semantics
    assert ntpath.isabs("C:\\foo") and ntpath.isabs("\\\\server\\share\\x")
    assert ntpath.splitdrive("\\\\?\\C:\\foo")[0] != ""
    assert PureWindowsPath("C:/data") / ".." / "x" != PureWindowsPath(
        "C:/x"
    )  # no normalising
    assert PureWindowsPath("A.pdf") == PureWindowsPath("a.pdf")  # case-insensitive
    assert PurePosixPath("A.pdf") != PurePosixPath("a.pdf")
    assert (
        download_name("CON") == "notebook"
        and download_name("Maths\\..\\x") == "Maths-x"
    )


def test_internal_storage_names_are_case_safe(settings):
    from litenotebook.reader.render import cache_dir

    names = [settings.source_file(1, i).name for i in range(1, 50)]
    assert all(n == n.lower() for n in names) and len({n.lower() for n in names}) == 49
    assert cache_dir(settings, 1, 2).name == cache_dir(settings, 1, 2).name.lower()


# --- files, handles, temp files -----------------------------------------------------


def test_pdf_handles_are_closed_after_every_operation(
    conn, settings, notebook, tmp_path, monkeypatch, store
):
    import pymupdf

    from conftest import FakeEmbedder, make_pdf
    from litenotebook.ingest.pipeline import delete_source, ingest_pdf
    from litenotebook.reader.annotations import chunk_target, open_source
    from litenotebook.reader.geometry import page_geometry
    from litenotebook.reader.render import render_page

    opened = []
    real_open = pymupdf.open

    def tracking(*a, **k):
        doc = real_open(*a, **k)
        opened.append(doc)
        return doc

    for module in (
        "pymupdf",
        "litenotebook.reader.render",
        "litenotebook.reader.geometry",
        "litenotebook.ingest.pdf",
    ):
        mod = sys.modules[module]
        if hasattr(mod, "open") and module == "pymupdf":
            monkeypatch.setattr(mod, "open", tracking)
    ingest_pdf(
        conn,
        settings,
        notebook,
        make_pdf(tmp_path / "h.pdf", ["One", "Two"]),
        FakeEmbedder(),
        store,
    )
    reader = open_source(conn, settings, notebook.id, 1)
    render_page(settings, reader, 1, 1.0)
    page_geometry(reader.path, 2)
    chunk = conn.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]
    chunk_target(conn, settings, notebook.id, chunk)
    assert opened and all(doc.is_closed for doc in opened), "a PDF was left open"
    delete_source(
        conn, settings, notebook.id, 1, store
    )  # works only if nothing holds it
    assert not settings.source_file(notebook.id, 1).exists()


def test_ocr_temp_files_are_closed_before_the_subprocess(monkeypatch, tmp_path):
    import litenotebook.ingest.pdf as pdf

    seen = {}

    def fake_run(args, **kwargs):
        image = Path(args[1])
        with open(image, "rb") as f:  # another process may open it: it is closed
            seen["bytes"] = len(f.read())
        seen["encoding"] = kwargs.get("encoding")
        seen["dir"] = image.parent
        return subprocess.CompletedProcess(args, 0, stdout="ocr text", stderr="")

    monkeypatch.setattr(pdf, "_tesseract", lambda: "tesseract")
    monkeypatch.setattr(pdf.subprocess, "run", fake_run)
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    assert pdf._ocr_page(page) == "ocr text"
    doc.close()
    assert seen["bytes"] > 0 and seen["encoding"] == "utf-8"
    assert not seen["dir"].exists()  # the temp directory is removed afterwards


def test_replace_and_remove_helpers(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"new")
    b.write_bytes(b"old")
    replace_file(a, b)
    assert b.read_bytes() == b"new" and not a.exists()
    tree = tmp_path / "tree" / "sub"
    tree.mkdir(parents=True)
    ro = tree / "readonly.txt"
    ro.write_text("x", encoding="utf-8")
    ro.chmod(0o444)  # Windows-style read-only attribute
    remove_tree(tmp_path / "tree")
    assert not (tmp_path / "tree").exists()


# --- SQLite -------------------------------------------------------------------------


def test_sqlite_features_and_concurrent_reads(settings):
    from litenotebook.db import open_db

    conn = open_db(settings.db_path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.execute("CREATE VIRTUAL TABLE temp.t USING fts5(x)")
    conn.execute(
        "INSERT INTO notebooks (name, embed_model, embed_dim) VALUES ('A', 'm', 4)"
    )
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO notebooks (name, embed_model, embed_dim) VALUES ('B', 'm', 4)"
    )
    reader = sqlite3.connect(settings.db_path)
    assert (
        reader.execute("SELECT count(*) FROM notebooks").fetchone()[0] == 1
    )  # WAL snapshot
    conn.commit()
    assert reader.execute("SELECT count(*) FROM notebooks").fetchone()[0] == 2
    reader.close()
    conn.close()


def test_restore_closes_every_connection_before_switching(tmp_path, monkeypatch):
    import litenotebook.archive.restore as restore
    from archive_fixture import build_rich_notebook
    from litenotebook.archive.export import create_backup
    from litenotebook.db import open_db

    src = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "src")})
    c = open_db(src.db_path)
    build_rich_notebook(c, src, tmp_path)
    backup = create_backup(c, src, tmp_path / "b.zip").path
    c.close()
    live = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "live")})
    open_db(live.db_path).close()

    opened = []
    real = restore.open_db

    def tracked(path):
        conn = real(path)
        opened.append(conn)
        return conn

    monkeypatch.setattr(restore, "open_db", tracked)
    restore.stage_restore(live, backup)
    for conn in opened:  # every staged connection was closed before the marker
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    from litenotebook import cli

    seen = {}
    real_apply = restore.apply_pending_restore

    def checking_apply(data_dir):
        with pytest.raises(sqlite3.ProgrammingError):
            seen["conn"].execute("SELECT 1")  # the CLI's own connection is closed too
        return real_apply(data_dir)

    real_open = cli.open_db

    def cli_open(path):
        seen["conn"] = real_open(path)
        return seen["conn"]

    restore.cancel_restore(live.data_dir)
    monkeypatch.setattr(cli, "open_db", cli_open)
    monkeypatch.setattr(restore, "apply_pending_restore", checking_apply)
    assert (
        cli.main(
            [
                "--data-dir",
                str(live.data_dir),
                "backup",
                "restore",
                str(backup),
                "--yes",
            ]
        )
        == 0
    )
    reopened = open_db(live.db_path)
    assert reopened.execute("SELECT count(*) FROM notebooks").fetchone()[0] == 1
    reopened.close()


def test_restore_recovers_from_an_interrupted_switch(tmp_path):
    from litenotebook.archive.restore import MARKER, apply_pending_restore, staging_dir

    data = tmp_path / "data"
    staging = staging_dir(data)
    safety = tmp_path / "data.pre-restore-20260101T000000Z"
    safety.mkdir()  # first rename happened …
    staging.mkdir()
    (staging / "litenotebook.sqlite3").write_bytes(b"")
    (staging / MARKER).write_text("{}", encoding="utf-8")  # … second did not
    applied = apply_pending_restore(data)
    assert applied is not None and data.is_dir() and not staging.exists()
    assert not (data / MARKER).exists() and safety.is_dir()
    (data / MARKER).write_text("{}", encoding="utf-8")  # crash after the switch
    assert apply_pending_restore(data) is None and not (data / MARKER).exists()


# --- system info and diagnostics ----------------------------------------------------


def test_system_info_is_portable_and_plausible(tmp_path):
    info = system_info(tmp_path)
    d = info.as_dict()
    for key in (
        "os",
        "architecture",
        "python_version",
        "logical_cpus",
        "total_ram_bytes",
        "available_ram_bytes",
        "disk_free_bytes",
        "cpu_name",
        "physical_cpus",
    ):
        assert key in d
    assert info.logical_cpus and info.logical_cpus >= 1
    assert info.disk_free_bytes and info.disk_free_bytes > 0
    if os_name() in ("linux", "windows"):
        assert info.total_ram_bytes and info.total_ram_bytes > 2**28
    assert system_info(
        tmp_path / "not" / "yet" / "created"
    ).disk_free_bytes  # parent used


def test_diagnostics_reports_without_secrets(tmp_path):
    from litenotebook.diagnostics import collect, format_text, redact_path, redact_url
    from litenotebook.web.app import create_app

    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    assert redact_path(Path.home() / "x" / "y") == "~" + os.sep + "x" + os.sep + "y"
    assert (
        redact_url("http://user:hunter2@127.0.0.1:8080/v1")
        == "http://127.0.0.1:8080/v1"
    )
    report = collect(
        s,
        effective_llm={
            "base_url": "http://u:sekret@h:1/v1",
            "model": "m",
            "api_key_configured": True,
        },
    )
    text = format_text(report)
    assert "sekret" not in text and "api_key_configured: True" in text
    # reports the truth (pytest itself may have imported torch for model tests;
    # the server process never does — checked in subprocess tests)
    assert report["libraries"]["torch_in_this_process"] == ("torch" in sys.modules)
    assert report["database"]["sqlite"]["fts5"] is True
    calls = []
    app = create_app(
        s,
        env={"LITENOTEBOOK_LLM_API_KEY": "sk-SECRET-VALUE"},
        embedder_factory=lambda: calls.append(1),
        llm_transport=httpx.MockTransport(
            lambda r: pytest.fail("diagnostics must not contact the LLM")
        ),
    )
    with TestClient(app, base_url=BASE) as c:
        page = c.get("/settings/diagnostics")
        assert page.status_code == 200 and "Copy diagnostics" in page.text
        assert (
            "sk-SECRET-VALUE" not in page.text
            and "api_key_configured: True" in page.text
        )
        assert 'data-copy-target="diag-text"' in page.text
    assert calls == []


# --- audits: the core stays OS-neutral ----------------------------------------------


CORE = [p for p in SRC.rglob("*.py") if "platform_utils" not in p.parts]
FORBIDDEN = (
    "/proc/",
    "/run/media",
    "shell=True",
    "os.fork",
    "xdg-open",
    "pkill",
    "taskkill",
    "systemctl",
    "pacman",
    "os.killpg",
    "signal.SIGKILL",
    "signal.SIGTERM",
    "fcntl",
    "msvcrt",
    "os.getuid",
    "os.chown",
    "NamedTemporaryFile",
    "start_new_session",
    "preexec_fn",
    "/home/",
    "C:\\\\Users",
    "os.system(",
    "os.popen(",
)


def test_core_has_no_os_specific_calls():
    offenders = []
    for path in CORE:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN:
            if needle in text:
                offenders.append((path.relative_to(SRC).as_posix(), needle))
    assert offenders == []


def test_core_text_io_always_names_its_encoding():
    offenders = []
    for path in [*CORE, *SRC.joinpath("platform_utils").glob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            keywords = {k.arg for k in node.keywords}
            if name in ("read_text", "write_text") and "encoding" not in keywords:
                offenders.append((path.name, node.lineno, name))
            if name == "open" and isinstance(node.func, ast.Name):
                mode = (
                    node.args[1].value
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    else "r"
                )
                if "b" not in str(mode) and "encoding" not in keywords:
                    offenders.append((path.name, node.lineno, "open"))
    assert offenders == []


def test_subprocess_only_through_argument_lists():
    for path in CORE:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
            ):
                first = node.args[0] if node.args else None
                assert not isinstance(first, ast.Constant), (path.name, node.lineno)


# --- cross-platform archive artefact ------------------------------------------------


def test_portable_fixture_archive_is_deterministic_and_valid(tmp_path):
    import hashlib
    import zipfile

    from litenotebook.archive.format import Limits
    from litenotebook.archive.reader import verify_archive
    from portable_archive import build

    a, b = build(tmp_path / "a.zip"), build(tmp_path / "b.zip")
    assert (
        hashlib.sha256(a.read_bytes()).digest()
        == hashlib.sha256(b.read_bytes()).digest()
    )
    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "d")})
    report = verify_archive(a, Limits.from_settings(s), current_schema=8)
    assert report.notebooks[0].meta["name"] == "Portable Ünïcode notes"
    with zipfile.ZipFile(a) as zf:
        for name in zf.namelist():
            assert archive_name_problem(name) is None and "\\" not in name
    assert a.stat().st_size < 64 * 1024  # small enough to copy around


def test_archive_case_collisions_still_rejected(tmp_path, conn, settings):
    from archive_fixture import build_rich_notebook, read_entries, write_archive
    from litenotebook.archive.export import export_notebook
    from litenotebook.archive.format import Limits
    from litenotebook.archive.reader import verify_archive

    rich = build_rich_notebook(conn, settings, tmp_path)
    entries = list(
        read_entries(
            export_notebook(conn, settings, rich.notebook_id, tmp_path / "x.zip").path
        ).items()
    )
    pdf = next(n for n, _ in entries if n.endswith(".pdf"))
    bad = write_archive(
        tmp_path / "case.zip", [*entries, (pdf.upper(), b"%PDF-")], fix_checksums=False
    )
    with pytest.raises(ArchiveError, match="Duplicate entry|Unexpected entry"):
        verify_archive(bad, Limits.from_settings(settings), current_schema=8)


def test_startup_latency_is_reasonable(tmp_path):
    from litenotebook.web.app import create_app

    s = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    started = time.perf_counter()
    app = create_app(
        s,
        env={},
        llm_transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": []})
        ),
    )
    with TestClient(app, base_url=BASE) as c:
        assert c.get("/").status_code == 200
    assert time.perf_counter() - started < 10
