"""Phase 3 web: PDF upload, background ingestion, status polling, retry, delete."""

import re
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.config import load_settings
from litenotebook.db import connect, open_db
from litenotebook.index.faiss_store import FaissStore
from litenotebook.ingest.pipeline import register_source
from litenotebook.notebooks import create_notebook, get_notebook
from litenotebook.sources import INTERRUPTED_MESSAGE, get_source, list_sources
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
SAME_ORIGIN = {"Origin": BASE}
HX = {**SAME_ORIGIN, "HX-Request": "true"}
PAGES = [
    "Gaussian elimination reduces a matrix to row echelon form using pivots.",
    "The rank of a matrix equals the number of pivots. Quillport is a harbour.",
]


class SwitchableEmbedder(FakeEmbedder):
    """Fails while ``fail`` is set (to exercise failure → retry)."""

    fail = False

    def encode(self, texts):
        if SwitchableEmbedder.fail:
            raise RuntimeError("embedding exploded")
        return super().encode(texts)


class CountingFactory:
    def __init__(self, cls=FakeEmbedder):
        self.cls, self.calls = cls, 0

    def __call__(self):
        self.calls += 1
        return self.cls()


@pytest.fixture
def app_settings(tmp_path):
    return load_settings(
        {
            "LITENOTEBOOK_DATA_DIR": str(tmp_path / "data"),
            "LITENOTEBOOK_MAX_UPLOAD_MB": "1",
        }
    )


@pytest.fixture
def factory():
    SwitchableEmbedder.fail = False
    return CountingFactory(SwitchableEmbedder)


def make_app(app_settings, factory):
    offline = httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("down", request=r))
    )
    return create_app(
        app_settings, env={}, llm_transport=offline, embedder_factory=factory
    )


@pytest.fixture
def app(app_settings, factory):
    return make_app(app_settings, factory)


@pytest.fixture
def client(app):
    with TestClient(app, base_url=BASE) as c:  # runs the lifespan (worker)
        yield c


@pytest.fixture
def nb(client):
    client.post("/notebooks", data={"name": "Research"}, headers=SAME_ORIGIN)
    return 1


@pytest.fixture
def pdf(tmp_path):
    return make_pdf(tmp_path / "linear.pdf", PAGES)


def upload(
    client,
    nb_id,
    data,
    filename="linear.pdf",
    mime="application/pdf",
    hx=True,
    ocr=False,
    headers=None,
):
    return client.post(
        f"/notebooks/{nb_id}/sources",
        files={"file": (filename, data, mime)},
        data={"ocr": "1"} if ocr else {},
        headers=headers if headers is not None else (HX if hx else SAME_ORIGIN),
        follow_redirects=False,
    )


def wait(app):
    assert app.state.worker.wait_idle(15), "ingestion did not finish"


def db(app_settings):
    return connect(app_settings.db_path)


def data_files(app_settings):
    root = app_settings.data_dir
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and not p.name.startswith("litenotebook.sqlite3")
        and p.name != ".instance.lock"  # the server's data-directory lock (Phase 10)
    )


# --- upload → background ingestion → ready --------------------------------------


def test_upload_ingests_in_background(client, app, app_settings, nb, pdf):
    r = upload(client, nb, pdf.read_bytes())
    assert r.status_code == 200
    assert "processing in the background" in r.text
    assert 'hx-swap-oob="afterbegin:#source-list"' in r.text
    assert 'id="source-empty" hidden' in r.text
    wait(app)

    status = client.get(f"/notebooks/{nb}/sources/1/status", headers=HX)
    assert status.status_code == 200
    assert "pill-ready" in status.text and "hx-trigger" not in status.text
    assert re.search(r"Pages</dt><dd>2</dd>", status.text)
    assert '<dd id="ws-chunks" hx-swap-oob="true">2</dd>' in status.text

    conn = db(app_settings)
    src = get_source(conn, nb, 1)
    assert src.status == "ready" and src.page_count == 2 and src.chunk_count == 2
    assert FaissStore(conn, app_settings).check(nb).ok
    conn.close()
    assert data_files(app_settings) == [
        "notebooks/1/index.faiss",
        "notebooks/1/index.meta.json",
        "notebooks/1/sources/1.pdf",
    ]
    page = client.get(f"/notebooks/{nb}?tab=sources").text
    assert "linear.pdf" in page and "pill-ready" in page


def test_plain_form_upload_redirects(client, app, nb, pdf):
    r = upload(client, nb, pdf.read_bytes(), hx=False)
    assert (
        r.status_code == 303 and r.headers["location"] == f"/notebooks/{nb}?tab=sources"
    )
    wait(app)


def test_active_source_row_polls_until_settled(app, app_settings, nb, pdf):
    conn = db(app_settings)
    src = register_source(conn, app_settings, get_notebook(conn, nb), pdf)
    conn.close()
    with TestClient(app, base_url=BASE) as c:  # second lifespan: not enqueued
        pass
    # recovery failed it (it was pending at startup); make it pending again
    conn = db(app_settings)
    conn.execute(
        "UPDATE sources SET status='pending', stage='uploaded' WHERE id=?", (src.id,)
    )
    conn.commit()
    conn.close()
    c = TestClient(app, base_url=BASE)  # no lifespan → no worker
    row = c.get(f"/notebooks/{nb}/sources/{src.id}/status", headers=HX).text
    assert 'hx-trigger="every 1s"' in row and "Uploaded · waiting" in row
    conn = db(app_settings)
    conn.execute(
        "UPDATE sources SET status='processing', stage='embedding', progress_done=3, "
        "progress_total=10 WHERE id=?",
        (src.id,),
    )
    conn.commit()
    conn.close()
    row = c.get(f"/notebooks/{nb}/sources/{src.id}/status", headers=HX).text
    assert "Embedding · 3/10 chunks" in row and '<progress max="10" value="3">' in row
    assert "Delete" not in row  # processing sources cannot be deleted


# --- size limits and streaming ----------------------------------------------------


def test_oversized_upload_rejected_by_content_length(client, app_settings, nb):
    big = b"%PDF-1.7\n" + b"0" * (2 * 1024 * 1024)
    r = upload(client, nb, big)
    assert r.status_code == 413 and "1 MiB upload limit" in r.text
    assert data_files(app_settings) == []
    conn = db(app_settings)
    assert list_sources(conn, nb) == []
    conn.close()


def test_oversized_chunked_upload_is_cut_off_while_streaming(client, app_settings, nb):
    """No Content-Length: the limit is enforced on the bytes actually received."""
    boundary = "LiteBoundary"
    sent = []

    def body():
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="big.pdf"\r\nContent-Type: application/pdf\r\n\r\n%PDF-1.7\n'
        ).encode()
        for _ in range(40):  # 40 × 64 KiB = 2.5 MiB
            sent.append(1)
            yield b"0" * 65536
        yield f"\r\n--{boundary}--\r\n".encode()

    r = client.post(
        f"/notebooks/{nb}/sources",
        content=body(),
        headers={**HX, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert r.status_code == 413
    assert data_files(app_settings) == []  # partial upload removed


def test_upload_streams_in_bounded_writes(client, app, nb, pdf, monkeypatch):
    """File bytes reach disk in buffer-sized writes as the body arrives."""
    from litenotebook.services import uploads

    writes = []
    real = uploads.run_in_threadpool

    async def spy(func, *args, **kwargs):
        if getattr(func, "__name__", "") == "write":
            writes.append(len(args[0]))
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(uploads, "run_in_threadpool", spy)
    monkeypatch.setattr(uploads, "WRITE_BUFFER_BYTES", 2048)
    data, boundary = pdf.read_bytes(), "LiteBoundary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="linear.pdf"\r\nContent-Type: application/pdf\r\n\r\n'
    ).encode()

    def body():  # 1 KiB network chunks, no Content-Length
        payload = head + data + f"\r\n--{boundary}--\r\n".encode()
        for i in range(0, len(payload), 1024):
            yield payload[i : i + 1024]

    r = client.post(
        f"/notebooks/{nb}/sources",
        content=body(),
        headers={**HX, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert r.status_code == 200, r.text
    wait(app)
    assert sum(writes) == len(data) and len(writes) >= len(data) // 3072
    assert max(writes) <= 2048 + 1024  # buffer + at most one network chunk


# --- validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        b"MZ\x90\x00 executable",
        b"<html><script>alert(1)</script></html>",
        b"PK\x03\x04zip",
    ],
)
def test_fake_pdf_rejected(client, app_settings, nb, payload):
    r = upload(client, nb, payload, filename="report.pdf")
    assert r.status_code == 415
    assert "report.pdf is not a PDF. Only PDF files are supported." in r.text
    assert data_files(app_settings) == []


def test_real_pdf_with_misleading_extension_and_mime(
    client, app, app_settings, nb, pdf
):
    r = upload(client, nb, pdf.read_bytes(), filename="notes.txt", mime="image/png")
    assert r.status_code == 200
    wait(app)
    conn = db(app_settings)
    src = get_source(conn, nb, 1)
    assert src.status == "ready" and src.mime == "application/pdf"
    assert src.original_filename == "notes.txt"
    conn.close()


def test_non_multipart_and_missing_file(client, nb, pdf):
    r = client.post(
        f"/notebooks/{nb}/sources",
        content=pdf.read_bytes(),
        headers={**HX, "Content-Type": "application/pdf"},
    )
    assert r.status_code == 415
    r = client.post(
        f"/notebooks/{nb}/sources",
        data={"ocr": "1"},
        files={"x": ("", b"")},
        headers=HX,
    )
    assert r.status_code == 400
    r = upload(client, nb, b"")
    assert r.status_code == 400 and "empty" in r.text


def test_upload_to_missing_notebook_is_404(client, pdf):
    assert upload(client, 999, pdf.read_bytes()).status_code == 404


@pytest.mark.parametrize(
    "filename",
    [
        "../../../../etc/passwd.pdf",
        "/etc/shadow.pdf",
        "..\\..\\windows\\system32\\evil.pdf",
        "a/../../b.pdf",
        "nul\x00byte.pdf",
        "ctrl\r\nchars.pdf",
        "..",
        "....//....//x.pdf",
        "‮gpj.pdf",
        "con.pdf",
    ],
)
def test_malicious_filenames_never_reach_the_filesystem(
    client, app, app_settings, nb, pdf, filename
):
    r = upload(client, nb, pdf.read_bytes(), filename=filename)
    assert r.status_code == 200, r.text
    wait(app)
    assert data_files(app_settings) == [
        "notebooks/1/index.faiss",
        "notebooks/1/index.meta.json",
        "notebooks/1/sources/1.pdf",
    ]
    conn = db(app_settings)
    shown = get_source(conn, nb, 1).original_filename
    conn.close()
    assert "/" not in shown and "\\" not in shown and "\x00" not in shown
    assert not any(ord(ch) < 32 for ch in shown)


def test_html_in_filenames_and_errors_is_escaped(client, app, nb, pdf):
    evil = '<img src=x onerror="alert(1)">.pdf'
    r = upload(client, nb, b"not a pdf", filename=evil)
    assert r.status_code == 415
    assert "<img src=x" not in r.text and "&lt;img src=x onerror=" in r.text

    assert upload(client, nb, pdf.read_bytes(), filename=evil).status_code == 200
    wait(app)
    for url in (
        f"/notebooks/{nb}?tab=sources",
        f"/notebooks/{nb}/sources/1",
        f"/notebooks/{nb}/sources/1/status",
    ):
        page = client.get(url, headers=HX if url.endswith("status") else {}).text
        assert "<img src=x" not in page and "&lt;img src=x onerror=" in page, url
    dup = upload(client, nb, pdf.read_bytes(), filename=evil)
    assert dup.status_code == 409 and "<img src=x" not in dup.text


# --- duplicates ---------------------------------------------------------------------


def test_duplicate_upload_same_notebook(client, app, app_settings, nb, pdf):
    assert upload(client, nb, pdf.read_bytes()).status_code == 200
    wait(app)
    r = upload(client, nb, pdf.read_bytes(), filename="copy.pdf")
    assert r.status_code == 409
    assert "copy.pdf is already in this notebook as “linear.pdf”" in r.text
    conn = db(app_settings)
    assert len(list_sources(conn, nb)) == 1
    conn.close()
    assert data_files(app_settings).count("notebooks/1/sources/1.pdf") == 1
    plain = upload(client, nb, pdf.read_bytes(), hx=False)
    assert plain.status_code == 409 and "already in this notebook" in plain.text


def test_same_pdf_in_different_notebooks(client, app, app_settings, nb, pdf):
    client.post("/notebooks", data={"name": "Other"}, headers=SAME_ORIGIN)
    assert upload(client, 1, pdf.read_bytes()).status_code == 200
    assert upload(client, 2, pdf.read_bytes()).status_code == 200
    wait(app)
    conn = db(app_settings)
    assert get_source(conn, 1, 1).status == get_source(conn, 2, 2).status == "ready"
    conn.close()


# --- isolation ---------------------------------------------------------------------


def test_cross_notebook_source_access_is_404(client, app, app_settings, nb, pdf):
    client.post("/notebooks", data={"name": "Other"}, headers=SAME_ORIGIN)
    upload(client, nb, pdf.read_bytes())
    wait(app)
    other = 2
    for method, url, data in [
        ("GET", f"/notebooks/{other}/sources/1", None),
        ("GET", f"/notebooks/{other}/sources/1/status", None),
        ("GET", f"/notebooks/{other}/sources/1/delete", None),
        ("POST", f"/notebooks/{other}/sources/1/retry", {}),
        ("POST", f"/notebooks/{other}/sources/1/delete", {"confirm": "1"}),
    ]:
        r = client.request(method, url, data=data, headers=HX)
        assert r.status_code == 404, (method, url)
    conn = db(app_settings)
    assert get_source(conn, nb, 1).status == "ready"
    conn.close()
    assert "linear.pdf" not in client.get(f"/notebooks/{other}?tab=sources").text


# --- delete --------------------------------------------------------------------------


def test_delete_flow_with_confirmation(client, app, app_settings, nb, pdf, tmp_path):
    upload(client, nb, pdf.read_bytes())
    keep = make_pdf(tmp_path / "keep.pdf", ["Determinants and cofactors."])
    upload(client, nb, keep.read_bytes(), filename="keep.pdf")
    wait(app)

    confirm = client.get(f"/notebooks/{nb}/sources/1/delete", headers=HX)
    assert confirm.status_code == 200
    assert "Delete this source permanently?" in confirm.text
    assert re.search(r"<dt>Source</dt><dd>linear.pdf</dd>", confirm.text)
    assert re.search(r"<dt>Chunks</dt><dd>2</dd>", confirm.text)

    plain = client.get(f"/notebooks/{nb}/sources/1/delete", follow_redirects=False)
    assert plain.status_code == 303 and plain.headers["location"].endswith(
        "?confirm=delete"
    )
    assert (
        "Delete this source permanently?" in client.get(plain.headers["location"]).text
    )

    unconfirmed = client.post(f"/notebooks/{nb}/sources/1/delete", headers=HX)
    assert unconfirmed.status_code == 422 and "not confirmed" in unconfirmed.text

    r = client.post(
        f"/notebooks/{nb}/sources/1/delete", data={"confirm": "1"}, headers=HX
    )
    assert r.status_code == 200 and "Deleted “linear.pdf”" in r.text
    assert '<dd id="ws-sources" hx-swap-oob="true">1</dd>' in r.text

    conn = db(app_settings)
    assert [s.id for s in list_sources(conn, nb)] == [2]
    assert (
        conn.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'Quillport'"
        ).fetchone()[0]
        == 0
    )
    status = FaissStore(conn, app_settings).check(nb)
    assert status.ok and status.index_count == get_source(conn, nb, 2).chunk_count
    conn.close()
    assert "notebooks/1/sources/1.pdf" not in data_files(app_settings)
    assert "notebooks/1/sources/2.pdf" in data_files(app_settings)
    assert client.get(f"/notebooks/{nb}/sources/1").status_code == 404


def test_cannot_delete_processing_source_or_its_notebook(app, app_settings, nb, pdf):
    conn = db(app_settings)
    src = register_source(conn, app_settings, get_notebook(conn, nb), pdf)
    conn.execute("UPDATE sources SET status='processing' WHERE id=?", (src.id,))
    conn.commit()
    conn.close()
    c = TestClient(app, base_url=BASE)  # no lifespan: nothing recovers it
    r = c.post(
        f"/notebooks/{nb}/sources/{src.id}/delete",
        data={"confirm": str(src.id)},
        headers=HX,
    )
    assert r.status_code == 409 and "being ingested" in r.text
    r = c.post(f"/notebooks/{nb}/delete", data={"confirm": str(nb)}, headers=HX)
    assert r.status_code == 409 and "being ingested" in r.text
    assert c.get(f"/notebooks/{nb}").status_code == 200


# --- failure & retry -------------------------------------------------------------


def test_failed_ingestion_then_retry(client, app, app_settings, nb, pdf):
    SwitchableEmbedder.fail = True
    upload(client, nb, pdf.read_bytes())
    wait(app)
    row = client.get(f"/notebooks/{nb}/sources/1/status", headers=HX).text
    assert "pill-failed" in row and "Retry" in row
    assert "Unexpected error while embedding" in row
    assert "embedding exploded" not in row and "Traceback" not in row
    conn = db(app_settings)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert FaissStore(conn, app_settings).check(nb).ok
    conn.close()

    SwitchableEmbedder.fail = False
    r = client.post(f"/notebooks/{nb}/sources/1/retry", headers=HX)
    assert r.status_code == 200
    wait(app)
    conn = db(app_settings)
    src = get_source(conn, nb, 1)
    assert src.status == "ready" and src.attempts == 2 and src.chunk_count == 2
    assert [s.id for s in list_sources(conn, nb)] == [1]
    assert FaissStore(conn, app_settings).check(nb).ok
    conn.close()
    again = client.post(f"/notebooks/{nb}/sources/1/retry", headers=HX)
    assert again.status_code == 409 and "Only failed sources" in again.text


def test_retry_plain_form_from_detail_page(client, app, nb, pdf):
    SwitchableEmbedder.fail = True
    upload(client, nb, pdf.read_bytes())
    wait(app)
    detail = client.get(f"/notebooks/{nb}/sources/1").text
    assert "Retry ingestion" in detail and "Unexpected error while embedding" in detail
    SwitchableEmbedder.fail = False
    r = client.post(
        f"/notebooks/{nb}/sources/1/retry", headers=SAME_ORIGIN, follow_redirects=False
    )
    assert r.status_code == 303
    wait(app)
    assert "pill-ready" in client.get(f"/notebooks/{nb}/sources/1").text


def test_scanned_pdf_failure_message(tmp_path, factory):
    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "big")})
    app = make_app(settings, factory)
    scan = make_pdf(tmp_path / "scan.pdf", ["SCANNED"], image_only={1})  # ~6 MB
    with TestClient(app, base_url=BASE) as c:
        c.post("/notebooks", data={"name": "n"}, headers=SAME_ORIGIN)
        assert upload(c, 1, scan.read_bytes(), filename="scan.pdf").status_code == 200
        wait(app)
        row = c.get("/notebooks/1/sources/1/status", headers=HX).text
    assert "No extractable text" in row and "OCR pages with no text layer" in row


def test_ocr_checkbox_propagates(client, app, app_settings, nb, pdf, tmp_path):
    upload(client, nb, pdf.read_bytes(), ocr=True)
    other = make_pdf(tmp_path / "o.pdf", ["A different page of text."])
    upload(client, nb, other.read_bytes(), filename="o.pdf")
    wait(app)
    conn = db(app_settings)
    assert get_source(conn, nb, 1).ocr is True and get_source(conn, nb, 2).ocr is False
    conn.close()
    assert "OCR on" in client.get(f"/notebooks/{nb}?tab=sources").text


# --- restart recovery -------------------------------------------------------------


def test_restart_marks_interrupted_sources_failed(app_settings, factory, pdf):
    conn = open_db(app_settings.db_path)
    notebook = create_notebook(conn, "Research")
    a = register_source(conn, app_settings, notebook, pdf)
    b = register_source(
        conn, app_settings, notebook, make_pdf(pdf.with_name("b.pdf"), ["Text b."])
    )
    conn.execute(
        "UPDATE sources SET status='processing', stage='embedding' WHERE id=?", (b.id,)
    )
    conn.commit()
    conn.close()
    partial = app_settings.upload_tmp_dir / "deadbeef.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"half an upload")

    app = make_app(app_settings, factory)
    with TestClient(app, base_url=BASE) as c:
        page = c.get(f"/notebooks/{notebook.id}?tab=sources").text
        assert page.count("pill-failed") == 2 and INTERRUPTED_MESSAGE in page
        assert app.state.worker.wait_idle(1)
    assert factory.calls == 0  # nothing re-ran on boot
    assert not partial.exists()
    conn = connect(app_settings.db_path)
    for sid in (a.id, b.id):
        assert get_source(conn, notebook.id, sid).error == INTERRUPTED_MESSAGE
    conn.close()


# --- security ----------------------------------------------------------------------


def test_cross_origin_upload_is_refused(client, app_settings, nb, pdf):
    r = upload(
        client,
        nb,
        pdf.read_bytes(),
        headers={"Origin": "http://evil.example", "HX-Request": "true"},
    )
    assert r.status_code == 403
    r = upload(client, nb, pdf.read_bytes(), headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    assert data_files(app_settings) == []


def test_foreign_host_upload_is_refused(client, nb, pdf):
    r = upload(client, nb, pdf.read_bytes(), headers={**HX, "Host": "evil.example"})
    assert r.status_code == 400


# --- laziness ---------------------------------------------------------------------


def test_embedder_created_only_by_first_job_and_reused(
    client, app, factory, nb, pdf, tmp_path
):
    client.get(f"/notebooks/{nb}?tab=sources")
    client.get("/settings")
    assert factory.calls == 0
    upload(client, nb, pdf.read_bytes())
    wait(app)
    upload(
        client,
        nb,
        make_pdf(tmp_path / "2.pdf", ["Second doc."]).read_bytes(),
        filename="2.pdf",
    )
    wait(app)
    assert factory.calls == 1


def test_source_browsing_and_upload_do_not_load_minilm_faiss_or_torch(tmp_path):
    pdf = make_pdf(tmp_path / "x.pdf", PAGES)
    code = f"""
import sys, httpx
from fastapi.testclient import TestClient
from litenotebook.config import load_settings
from litenotebook.web.app import create_app
settings = load_settings({{"LITENOTEBOOK_DATA_DIR": {str(tmp_path / "d")!r}}})
fake = httpx.MockTransport(lambda r: httpx.Response(200, json={{"data": []}}))
app = create_app(settings, env={{}}, llm_transport=fake)
c = TestClient(app, base_url="{BASE}")   # no lifespan: the worker never runs
h = {{"Origin": "{BASE}"}}
c.post("/notebooks", data={{"name": "n"}}, headers=h)
data = open({str(pdf)!r}, "rb").read()
r = c.post("/notebooks/1/sources", headers=h, follow_redirects=False,
           files={{"file": ("x.pdf", data, "application/pdf")}})
assert r.status_code == 303, r.text
for p in ("/", "/notebooks/1", "/notebooks/1?tab=sources", "/notebooks/1/sources/1",
          "/notebooks/1/sources/1/status", "/settings", "/healthz"):
    assert c.get(p).status_code == 200, p
heavy = [m for m in ("torch", "sentence_transformers", "transformers", "faiss")
         if m in sys.modules]
print("HEAVY", heavy)
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert "HEAVY []" in out.stdout
