"""Log privacy (Phase 16): what the log file may and may not contain."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from conftest import FakeEmbedder, make_pdf
from litenotebook.applog import configure_file_logging, redact
from litenotebook.config import load_settings
from litenotebook.web.app import create_app

BASE = "http://127.0.0.1:5056"
ORIGIN = {"Origin": BASE}


def _flush() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_redact_masks_secrets_url_credentials_and_home():
    home = str(Path.home())
    text = redact(
        f"key sk-ABCDEF123456 Authorization: Bearer eyJabc.def.ghi "
        f"https://alice:s3cr3t-pass@proxy.example:8080/x "
        f"api_key=hunter2hunter2 password: pa55word "
        f"file {home}/Documents/Taxes 2025.pdf"
    )
    for secret in (
        "sk-ABCDEF123456",
        "eyJabc.def.ghi",
        "s3cr3t-pass",
        "hunter2hunter2",
        "pa55word",
        home,
    ):
        assert secret not in text, secret
    assert "https://alice:[redacted]@proxy.example" in text
    assert "~/Documents/Taxes 2025.pdf" in text


def test_tracebacks_are_redacted_too(tmp_path):
    settings = load_settings(
        {
            "LITENOTEBOOK_DATA_DIR": str(tmp_path / "d"),
            "LITENOTEBOOK_LOG_DIR": str(tmp_path / "logs"),
        }
    )
    path = configure_file_logging(settings)
    log = logging.getLogger("litenotebook.test")
    try:
        raise RuntimeError(f"upstream said api_key=TOPSECRET99 at {Path.home()}/x")
    except RuntimeError:
        log.exception("Request failed")
    _flush()
    text = path.read_text("utf-8")
    assert "Traceback" in text  # still diagnosable
    assert "TOPSECRET99" not in text and str(Path.home()) + "/x" not in text


def test_normal_use_never_logs_document_note_chat_or_search_text(tmp_path):
    settings = load_settings(
        {
            "LITENOTEBOOK_DATA_DIR": str(tmp_path / "data"),
            "LITENOTEBOOK_LOG_DIR": str(tmp_path / "logs"),
        }
    )
    path = configure_file_logging(settings, verbose=True)  # even at DEBUG

    def llm(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(("/props", "/models")):
            return httpx.Response(
                200, json={"data": [], "default_generation_settings": {}}
            )
        return httpx.Response(500, text="model crashed on PRIVATE-QUESTION-TEXT")

    app = create_app(
        settings,
        env={},
        llm_transport=httpx.MockTransport(llm),
        embedder_factory=FakeEmbedder,
    )
    pdf = make_pdf(
        tmp_path / "Private Medical Letter.pdf",
        ["CONFIDENTIAL-DOC-TEXT the patient record body."],
    )
    with TestClient(app, base_url=BASE) as c:
        c.post("/notebooks", data={"name": "Private"}, headers=ORIGIN)
        nb = 1
        c.post(
            f"/notebooks/{nb}/notes",
            data={"title": "Diary", "content": "SECRET-NOTE-BODY about my day"},
            headers=ORIGIN,
        )
        with open(pdf, "rb") as f:
            c.post(
                f"/notebooks/{nb}/sources",
                files={"file": (pdf.name, f, "application/pdf")},
                headers=ORIGIN,
            )
        c.get(f"/notebooks/{nb}/search", params={"q": "PRIVATE-SEARCH-TERM"})
        c.post(
            f"/notebooks/{nb}/chat/send",
            data={"question": "PRIVATE-QUESTION-TEXT?", "conversation_id": ""},
            headers=ORIGIN,
        )
        c.post("/settings", data={"llm_api_key": "sk-USERKEY123456789"}, headers=ORIGIN)
    _flush()
    text = path.read_text("utf-8")
    for private in (
        "CONFIDENTIAL-DOC-TEXT",
        "SECRET-NOTE-BODY",
        "PRIVATE-SEARCH-TERM",
        "PRIVATE-QUESTION-TEXT",
        "sk-USERKEY123456789",
    ):
        assert private not in text, private
    assert str(tmp_path) not in text or str(Path.home()) not in str(tmp_path)
