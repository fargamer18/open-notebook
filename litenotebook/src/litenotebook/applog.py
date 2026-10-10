"""Per-user log files for the desktop app (and ``serve --log-file``).

One rotating file, ``litenotebook.log`` (5 × 1 MiB), in the platform log
folder (``Settings.log_path``). Web access logs are not written: request
URLs can carry search text. Everything written — messages and tracebacks —
passes through ``redact``: API keys, bearer tokens, URL credentials and the
user's home folder (shown as ``~``) are masked. Document text, note bodies and
prompts are never logged by LiteNotebook.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
from pathlib import Path

from litenotebook.config import Settings

LOG_FILE = "litenotebook.log"
_SECRETS = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}|\b(sk-[A-Za-z0-9_-]{8,})"
    r"|((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s&\"']+"
)
# user:password@ in URLs (proxy settings, mirrors)
_URL_CREDENTIALS = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^/\s:@]+:)[^@\s/]+@")


def _home_variants() -> list[str]:
    try:
        home = str(Path.home())
    except (RuntimeError, OSError):
        return []
    if len(home) < 3:  # "/" or a drive root: nothing private to hide
        return []
    return sorted({home, home.replace("\\", "/")}, key=len, reverse=True)


def redact(text: str) -> str:
    """Mask secrets, URL credentials and the user's home folder."""
    text = _SECRETS.sub(lambda m: (m.group(1) or m.group(3) or "") + "[redacted]", text)
    text = _URL_CREDENTIALS.sub(r"\1[redacted]@", text)
    for home in _home_variants():
        flags = re.IGNORECASE if os.name == "nt" else 0
        text = re.sub(re.escape(home), "~", text, flags=flags)
    return text


class RedactSecrets(logging.Filter):
    """Masks the message itself (kept for handlers without our formatter)."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


class RedactingFormatter(logging.Formatter):
    """Masks the complete output, tracebacks included."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def configure_file_logging(settings: Settings, *, verbose: bool = False) -> Path:
    folder = settings.log_path
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOG_FILE
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(
        RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.addFilter(RedactSecrets())
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for old in list(root.handlers):
        if isinstance(old, logging.handlers.RotatingFileHandler):
            root.removeHandler(old)
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").disabled = True
    # httpx logs every request URL at INFO; URLs can carry query text
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path
