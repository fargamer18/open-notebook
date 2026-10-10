"""Low disk space and permission failures (Phase 16).

The disk is never actually filled: the OS error is injected where the write
happens. Each case must fail safely (no half-valid database rows, no partial
files) with a message the user can act on.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
from collections import namedtuple
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.ai.download import Downloader, InsufficientSpace
from litenotebook.config import load_settings
from litenotebook.platform_utils.filesystem import DISK_FULL_MESSAGE, is_disk_full
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
ORIGIN = {"Origin": BASE}
Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def web(tmp_path):  # type: ignore[no-untyped-def]
    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    app = create_app(
        settings,
        env={},
        llm_transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": []})
        ),
        embedder_factory=FakeEmbedder,
    )
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as c:
        c.post("/notebooks", data={"name": "Disk"}, headers=ORIGIN)
        yield c, settings


def _sources(settings) -> int:  # type: ignore[no-untyped-def]
    import sqlite3

    c = sqlite3.connect(settings.db_path)
    try:
        return int(c.execute("SELECT count(*) FROM sources").fetchone()[0])
    finally:
        c.close()


def test_disk_full_detection_covers_posix_windows_and_sqlite():
    import sqlite3

    assert is_disk_full(OSError(errno.ENOSPC, "No space left on device"))
    winerr = OSError(errno.EINVAL, "disk full")
    winerr.winerror = 112  # type: ignore[attr-defined]
    assert is_disk_full(winerr)
    assert is_disk_full(sqlite3.OperationalError("database or disk is full"))
    wrapped = RuntimeError("import failed")
    wrapped.__cause__ = OSError(errno.ENOSPC, "full")
    assert is_disk_full(wrapped)
    assert not is_disk_full(OSError(errno.EACCES, "Permission denied"))


def test_upload_refused_up_front_when_there_is_no_room(web, tmp_path, monkeypatch):
    c, settings = web
    from litenotebook.services import uploads

    monkeypatch.setattr(
        uploads.shutil, "disk_usage", lambda p: Usage(10**9, 10**9, 1000)
    )
    pdf = make_pdf(tmp_path / "a.pdf", ["Velt river"])
    with open(pdf, "rb") as f:
        r = c.post(
            "/notebooks/1/sources",
            files={"file": ("a.pdf", f, "application/pdf")},
            headers={**ORIGIN, "Accept": "text/html"},
        )
    assert r.status_code == 507 and "Not enough disk space" in r.text
    assert _sources(settings) == 0
    assert (
        not list(settings.upload_tmp_dir.glob("*"))
        if settings.upload_tmp_dir.exists()
        else True
    )


def test_upload_disk_full_while_writing_leaves_nothing(web, tmp_path, monkeypatch):
    c, settings = web
    from litenotebook.services import uploads

    real = uploads.create_exclusive

    class Full:
        def __init__(self, f):  # type: ignore[no-untyped-def]
            self.f = f

        def write(self, data):  # type: ignore[no-untyped-def]
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self):  # type: ignore[no-untyped-def]
            self.f.close()

    monkeypatch.setattr(uploads, "create_exclusive", lambda p: Full(real(p)))
    pdf = make_pdf(tmp_path / "a.pdf", ["Velt river " * 50])
    with open(pdf, "rb") as f:
        r = c.post(
            "/notebooks/1/sources",
            files={"file": ("a.pdf", f, "application/pdf")},
            headers=ORIGIN,
        )
    assert r.status_code == 507 and DISK_FULL_MESSAGE[:20] in r.text
    assert _sources(settings) == 0
    assert not list(settings.upload_tmp_dir.glob("*"))


def test_backup_with_a_full_disk_shows_a_clear_page(web, monkeypatch):
    c, settings = web
    from litenotebook.archive import export as export_mod

    def full(*a, **k):  # type: ignore[no-untyped-def]
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(export_mod, "_write_notebook", full)
    r = c.post(
        "/notebooks/1/export", data={}, headers={**ORIGIN, "Accept": "text/html"}
    )
    assert r.status_code == 507 and "Not enough disk space" in r.text
    assert "Traceback" not in r.text
    assert (
        not list(settings.archive_tmp_dir.glob("*.zip"))
        if settings.archive_tmp_dir.exists()
        else True
    )


def test_model_download_checks_space_before_fetching(tmp_path, monkeypatch):
    from litenotebook.ai import download

    monkeypatch.setattr(
        download.shutil, "disk_usage", lambda p: Usage(10**12, 10**12, 10**6)
    )
    requested: list[str] = []
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: requested.append(str(r.url)) or httpx.Response(200, content=b"x")
        )
    )
    with pytest.raises(InsufficientSpace):
        Downloader(client).fetch(
            "https://huggingface.co/x/y/resolve/abc/model.gguf",
            tmp_path / "model.gguf",
            sha256="0" * 64,
            size=4 * 1024**3,
        )
    assert requested == [] and not list(tmp_path.iterdir())


@pytest.mark.posix_only
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes"
)
def test_unwritable_page_cache_still_shows_the_page(
    conn, settings, notebook, embedder, store, tmp_path
):
    from litenotebook.ingest.pipeline import ingest_pdf
    from litenotebook.reader.render import (
        RenderCacheUnavailable,
        open_source,
        render_page,
    )

    report = ingest_pdf(
        conn,
        settings,
        notebook,
        make_pdf(tmp_path / "r.pdf", ["Readable page"]),
        embedder,
        store,
    )
    cache_root = settings.notebook_dir(notebook.id) / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        reader = open_source(conn, settings, notebook.id, report.source_id)
        with pytest.raises(RenderCacheUnavailable) as caught:
            render_page(settings, reader, 1, 1.0)
        assert caught.value.png[:4] == b"\x89PNG"
    finally:
        cache_root.chmod(stat.S_IRWXU)


@pytest.mark.posix_only
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes"
)
def test_unwritable_data_folder_gives_an_actionable_start_error(tmp_path, monkeypatch):
    import litenotebook.desktop as desktop

    data = tmp_path / "locked"
    data.mkdir()
    data.chmod(stat.S_IRUSR | stat.S_IXUSR)
    logs = tmp_path / "logs-locked"
    logs.mkdir()
    logs.chmod(stat.S_IRUSR | stat.S_IXUSR)
    shown: list[str] = []
    monkeypatch.setattr(
        desktop, "show_error_dialog", lambda t, m: shown.append(m) or True
    )
    settings = load_settings(
        {"LITENOTEBOOK_DATA_DIR": str(data), "LITENOTEBOOK_LOG_DIR": str(logs)}
    )
    try:
        assert desktop.run(["--no-browser"], settings=settings) == 1
    finally:
        data.chmod(stat.S_IRWXU)
        logs.chmod(stat.S_IRWXU)
    assert shown and "cannot write to its data folder" in shown[0]
    assert "--data-dir" in shown[0]


def test_missing_source_file_gives_a_clear_reader_error(web, tmp_path):
    c, settings = web
    import sqlite3

    from litenotebook.index.faiss_store import FaissStore
    from litenotebook.ingest.pipeline import ingest_pdf
    from litenotebook.notebooks import get_notebook

    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        report = ingest_pdf(
            conn,
            settings,
            get_notebook(conn, 1),
            make_pdf(tmp_path / "gone.pdf", ["Vanishing page"]),
            FakeEmbedder(),
            FaissStore(conn, settings),
        )
    finally:
        conn.close()
    settings.source_file(1, report.source_id).unlink()  # removed outside the app
    r = c.get(f"/notebooks/1/sources/{report.source_id}/pages/1/render")
    assert r.status_code == 404 and "not available" in r.json()["error"]
    page = c.get(
        f"/notebooks/1/sources/{report.source_id}/read", headers={"Accept": "text/html"}
    )
    assert page.status_code in (200, 404) and "Traceback" not in page.text
    assert c.get("/notebooks/1/search", params={"q": "Vanishing"}).status_code == 200


def test_missing_embedding_runtime_is_reported_not_crashing(tmp_path, monkeypatch):
    """Frozen app, runtime folder deleted after install: a clear message."""
    from litenotebook import packaging
    from litenotebook.packaging import (
        EmbeddingRuntimeMissingError,
        embed_worker_command,
    )

    monkeypatch.setattr(packaging, "is_frozen", lambda: True)
    monkeypatch.setattr(packaging, "app_dir", lambda: tmp_path / "app")
    monkeypatch.delenv("LITENOTEBOOK_EMBED_WORKER", raising=False)
    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    with pytest.raises(EmbeddingRuntimeMissingError, match="not installed"):
        embed_worker_command(settings)


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX path semantics in this check"
)
def test_tmp_and_cache_paths_live_inside_the_data_folder(tmp_path):
    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "d")})
    for p in (settings.upload_tmp_dir, settings.archive_tmp_dir):
        assert Path(p).resolve().is_relative_to((tmp_path / "d").resolve())
