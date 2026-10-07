import pytest

from conftest import make_pdf
from litenotebook.ingest.pdf import (
    PdfError,
    extract_pages,
    normalize_page_text,
    tesseract_available,
)


def test_extracts_each_page_with_one_based_numbers(tmp_path):
    pdf = make_pdf(
        tmp_path / "doc.pdf",
        [
            "Alpha page talks about rank.",
            "Beta page covers pivots.",
            "Gamma page ends.",
        ],
    )
    pages = extract_pages(pdf)
    assert [p.page_no for p in pages] == [1, 2, 3]
    assert (
        "Alpha" in pages[0].text
        and "Beta" in pages[1].text
        and "Gamma" in pages[2].text
    )
    assert "Beta" not in pages[0].text
    assert all(p.method == "text" for p in pages)


def test_blank_page_is_kept_and_marked_empty(tmp_path):
    pages = extract_pages(
        make_pdf(tmp_path / "d.pdf", ["First page text here.", "", "Third."])
    )
    assert [p.page_no for p in pages] == [1, 2, 3]
    assert pages[1].method == "empty" and pages[1].text == ""


def test_rejects_non_pdf(tmp_path):
    bogus = tmp_path / "x.pdf"
    bogus.write_text("not a pdf")
    with pytest.raises(PdfError):
        extract_pages(bogus)


def test_normalize_page_text():
    raw = "Gauss-\nian  elimi­nation\r\n\r\n\r\n\nNext   para  \n line"
    assert normalize_page_text(raw) == "Gaussian elimination\n\nNext para\nline"


@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
def test_ocr_fallback_only_for_pages_without_text_layer(tmp_path):
    pdf = make_pdf(
        tmp_path / "scan.pdf",
        ["Typed page with a real text layer.", "SCANNED PAGE ABOUT EIGENVALUES"],
        image_only={2},
    )
    assert extract_pages(pdf)[1].method == "empty"  # OCR is opt-in
    pages = extract_pages(pdf, ocr=True)
    assert pages[0].method == "text"
    assert pages[1].method == "ocr"
    assert "EIGENVALUES" in pages[1].text.upper()
