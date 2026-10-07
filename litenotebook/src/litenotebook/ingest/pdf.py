"""Page-by-page PDF text extraction with PyMuPDF.

Pages keep their 1-based PDF page number. Pages with no usable text layer can
optionally be OCR'd with Tesseract (ported from LocalBook's ocr_index.py, but
applied only where the text layer is empty).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf

MIN_TEXT_CHARS = 20  # below this a page counts as having no text layer
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


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _ocr_page(page: pymupdf.Page) -> str:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(OCR_ZOOM, OCR_ZOOM), alpha=False)
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        pix.save(tmp.name)
        result = subprocess.run(
            ["tesseract", tmp.name, "stdout", "-l", "eng", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    return result.stdout if result.returncode == 0 else ""


def extract_pages(path: Path, *, ocr: bool = False) -> list[PageText]:
    """Extract every page of ``path``. With ``ocr=True``, empty pages are OCR'd."""
    try:
        doc = pymupdf.open(path)
    except Exception as e:  # PyMuPDF raises several exception types
        raise PdfError(f"Cannot open PDF {path.name}: {e}") from e
    with doc:
        if not doc.is_pdf:
            raise PdfError(f"{path.name} is not a PDF")
        if doc.needs_pass:
            raise PdfError(f"{path.name} is password-protected")
        use_ocr = ocr and tesseract_available()
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
    return pages
