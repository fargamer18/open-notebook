"""LiteNotebook's PDF engine boundary.

Everything that reads PDFs goes through this package: upload validation,
page count, page text for ingestion, word boxes for highlights, page images
for the Reader and OCR. The rest of the application never imports a PDF
library directly. The implementation is PDFium (via pypdfium2); see
``docs/pdf-engine.md`` for the coordinate convention and known differences
from the previous engine (PyMuPDF, removed in Phase 17).
"""

from litenotebook.pdf.engine import PDF_LOCK, engine_versions, open_pdf
from litenotebook.pdf.models import (
    PageSize,
    PdfDamagedError,
    PdfError,
    PdfNoPagesError,
    PdfPasswordError,
    RenderedPage,
    WordBox,
)

__all__ = [
    "PDF_LOCK",
    "PageSize",
    "PdfDamagedError",
    "PdfError",
    "PdfNoPagesError",
    "PdfPasswordError",
    "RenderedPage",
    "WordBox",
    "engine_versions",
    "open_pdf",
]
