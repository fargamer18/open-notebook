"""Page-by-page PDF text extraction with PyMuPDF.

Pages keep their 1-based PDF page number. Pages with no usable text layer can
optionally be OCR'd with Tesseract (ported from LocalBook's ocr_index.py, but
applied only where the text layer is empty).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf

MIN_TEXT_CHARS = 20  # below this a page counts as having no text layer
PDF_MAGIC = b"%PDF-"
MAGIC_WINDOW = 1024  # PDF readers accept the header within the first 1 KiB
OCR_ZOOM = 2.0

PageMethod = Literal["text", "ocr", "empty"]


class PdfError(ValueError):
    pass


@dataclass(frozen=True)
class PageText:
    page_no: int  # 1-based
    text: str
    method: PageMethod


def normalize_page_text(text: str) -> str:
    """Light cleanup that keeps paragraph structure (chunk offsets index into this)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tesseract() -> str | None:
    from litenotebook.platform_utils.system import find_executable

    found = find_executable("tesseract")  # tesseract.exe on Windows
    return str(found) if found else None


def tesseract_available() -> bool:
    return _tesseract() is not None


def _ocr_page(page: pymupdf.Page) -> str:
    from litenotebook.platform_utils.processes import creation_flags

    exe = _tesseract()
    if exe is None:
        return ""
    pix = page.get_pixmap(matrix=pymupdf.Matrix(OCR_ZOOM, OCR_ZOOM), alpha=False)
    # A temporary directory rather than an open temp file: Windows will not let
    # tesseract open a file this process still holds open. The image is
    # written and closed before tesseract reads it; the directory is removed.
    with tempfile.TemporaryDirectory(prefix="litenotebook-ocr-") as tmp:
        image = Path(tmp) / "page.png"
        pix.save(str(image))
        result = subprocess.run(  # noqa: S603 - argument list, no shell
            [exe, str(image), "stdout", "-l", "eng", "--psm", "6"],
            capture_output=True,
            encoding="utf-8",  # tesseract writes UTF-8; never the locale codec
            errors="replace",
            timeout=120,
            check=False,
            creationflags=creation_flags(),
        )
    return result.stdout if result.returncode == 0 else ""


def has_pdf_header(path: Path) -> bool:
    """Cheap content sniff; never trusts the filename extension or MIME type."""
    with open(path, "rb") as f:
        return PDF_MAGIC in f.read(MAGIC_WINDOW)


def _open(path: Path, name: str) -> pymupdf.Document:
    """Open and validate a PDF. Error messages name ``name``, never the path."""
    if not has_pdf_header(path):
        raise PdfError(f"{name} is not a PDF (no %PDF- header)")
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as e:  # PyMuPDF raises several exception types
        raise PdfError(f"{name} is damaged or not a valid PDF") from e
    if not doc.is_pdf:
        doc.close()
        raise PdfError(f"{name} is not a PDF")
    if doc.needs_pass:
        doc.close()
        raise PdfError(f"{name} is password-protected")
    if doc.page_count < 1:
        doc.close()
        raise PdfError(f"{name} has no pages")
    return doc


def inspect_pdf(path: Path, *, name: str | None = None) -> int:
    """Validate that ``path`` really is a readable PDF; return its page count."""
    with _open(path, name or path.name) as doc:
        return int(doc.page_count)


def extract_pages(
    path: Path,
    *,
    ocr: bool = False,
    name: str | None = None,
    on_page: Callable[[int, int], None] | None = None,
) -> list[PageText]:
    """Extract every page of ``path``. With ``ocr=True``, empty pages are OCR'd.

    ``on_page(done, total)`` is called after each page.
    """
    with _open(path, name or path.name) as doc:
        use_ocr = ocr and tesseract_available()
        total = int(doc.page_count)
        pages: list[PageText] = []
        for index, page in enumerate(doc):
            text = normalize_page_text(page.get_text("text"))
            method: PageMethod = "text"
            if len(text) < MIN_TEXT_CHARS:
                ocr_text = normalize_page_text(_ocr_page(page)) if use_ocr else ""
                if len(ocr_text) > len(text):
                    text, method = ocr_text, "ocr"
                elif not text:
                    method = "empty"
            pages.append(PageText(page_no=index + 1, text=text, method=method))
            if on_page is not None:
                on_page(index + 1, total)
    return pages
