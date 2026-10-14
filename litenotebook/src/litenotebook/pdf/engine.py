"""Opening PDFs, and the one rule every caller must follow.

PDFium is not thread-safe: "It is not allowed to call pdfium functions
simultaneously across different threads, not even with different documents"
(pypdfium2 documentation). LiteNotebook calls it from request threads
(Reader images, word geometry) while ingestion runs in a background thread,
so every PDFium call — including closing objects — holds ``PDF_LOCK``.
The document class takes the lock itself, per operation, so a long
ingestion never blocks the Reader for more than one page's work. Objects are
always closed explicitly inside the lock, so pypdfium2's garbage-collection
finalizers never run PDFium code outside it.
"""

from __future__ import annotations

import threading
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litenotebook.pdf.pdfium_engine import PdfiumDocument

PDF_LOCK = threading.RLock()


def open_pdf(path: Path, name: str | None = None) -> PdfiumDocument:
    """Open and validate a PDF (raises ``PdfError`` subclasses with messages
    that name ``name``, never the path). Use as a context manager."""
    from litenotebook.pdf.pdfium_engine import PdfiumDocument

    return PdfiumDocument(Path(path), name or Path(path).name)


def engine_versions() -> dict[str, str]:
    """For Diagnostics: the wrapper and the PDFium build."""
    try:
        import pypdfium2

        return {
            "engine": "PDFium",
            "pypdfium2": str(pypdfium2.PYPDFIUM_INFO),
            "pdfium": str(pypdfium2.PDFIUM_INFO),
        }
    except Exception:  # pragma: no cover - reported, never fatal
        try:
            version = metadata.version("pypdfium2")
        except metadata.PackageNotFoundError:
            version = "missing"
        return {"engine": "PDFium", "pypdfium2": version, "pdfium": "unknown"}
