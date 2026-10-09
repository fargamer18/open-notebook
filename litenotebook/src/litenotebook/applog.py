"""Per-user log files for the desktop app (and ``serve --log-file``).

One rotating file, ``litenotebook.log`` (5 × 1 MiB), in the platform log
folder (``Settings.log_path``). Web access logs are not written: request
URLs can carry search text. A filter masks anything that looks like an API
key or bearer token. Document text is never logged by LiteNotebook.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
from pathlib import Path

from litenotebook.config import Settings

LOG_FILE = "litenotebook.log"
_SECRETS = re.compile(
    r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}|\b(sk-[A-Za-z0-9_-]{8,})"
    r"|((?:api[_-]?key|token|password)\s*[=:]\s*)[^\s&\"']+"
)


class RedactSecrets(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = _SECRETS.sub(
            lambda m: (m.group(1) or m.group(3) or "") + "[redacted]", message
        )
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


def configure_file_logging(settings: Settings, *, verbose: bool = False) -> Path:
    folder = settings.log_path
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOG_FILE
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.addFilter(RedactSecrets())
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for old in list(root.handlers):
        if isinstance(old, logging.handlers.RotatingFileHandler):
            root.removeHandler(old)
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").disabled = True
    return path
