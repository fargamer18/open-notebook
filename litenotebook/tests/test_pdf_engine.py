"""The PDF engine boundary (Phase 17: PDFium via pypdfium2).

Covers opening and validation, page sizes under rotation and crop boxes,
rendering orientation, word reconstruction, the coordinate convention that
stored annotations and citation highlights rely on, and that nothing needs
PyMuPDF any more.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from conftest import make_pdf
from litenotebook.pdf import (
    PdfDamagedError,
    PdfError,
    PdfNoPagesError,
    PdfPasswordError,
    engine_versions,
    open_pdf,
)
from litenotebook.pdf.png import encode_png
from litenotebook.reader.geometry import PageGeometry, Word, match_text
from pdfgen import Document, text_width

ROOT = Path(__file__).resolve().parents[1]


def _decode_png(data: bytes) -> tuple[int, int, int, bytes]:
    """Decode the engine's own PNGs (8-bit, filter 0): width, height, channels."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, header = 8, b"", b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            header = body
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    width, height, depth, color = struct.unpack(">IIBB", header[:10])
    assert depth == 8
    channels = {0: 1, 2: 3}[color]
    raw = zlib.decompress(idat)
    row = width * channels
    pixels = b"".join(
        raw[y * (row + 1) + 1 : (y + 1) * (row + 1)] for y in range(height)
    )
    assert all(raw[y * (row + 1)] == 0 for y in range(height))
    return width, height, channels, pixels


def _ink_box(data: bytes) -> tuple[float, float, float, float]:
    """Normalised bounding box of dark pixels in a rendered page."""
    width, height, channels, px = _decode_png(data)
    xs, ys = [], []
    for y in range(height):
        base = y * width * channels
        for x in range(width):
            if px[base + x * channels] < 100:
                xs.append(x)
                ys.append(y)
    assert xs, "nothing was drawn"
    return (
        min(xs) / width,
        min(ys) / height,
        (max(xs) + 1) / width,
        (max(ys) + 1) / height,
    )


def _display(
    rotate: int, x: float, y: float, w: float, h: float
) -> tuple[float, float]:
    """Where user-space point (x, y) of an unrotated w×h page appears on the
    displayed page, normalised, top-left origin (the documented convention)."""
    return {
        0: (x / w, 1 - y / h),
        90: (y / h, x / w),
        180: (1 - x / w, y / h),
        270: (1 - y / h, 1 - x / w),
    }[rotate]


# --- opening and validation ---------------------------------------------------------


def test_open_close_count_and_sizes(tmp_path):
    path = make_pdf(tmp_path / "a.pdf", ["first page", "second page", "third"])
    with open_pdf(path) as doc:
        assert doc.page_count == 3
        size = doc.page_size(0)
        assert (size.width, size.height, size.rotation) == (595, 842, 0)
        assert "second page" in doc.page_text(1)
    with pytest.raises(PdfError, match="closed"):
        doc.page_text(0)
    doc.close()  # closing twice is harmless


def test_page_index_is_checked(tmp_path):
    with open_pdf(make_pdf(tmp_path / "a.pdf", ["one"])) as doc:
        for bad in (-1, 1, 99):
            with pytest.raises(IndexError):
                doc.page_size(bad)


@pytest.mark.parametrize(
    "payload",
    [b"%PDF-1.7\nnot really\n", b"%PDF-1.4\n" + b"\x00" * 2048, b"garbage" * 50],
)
def test_damaged_input_is_rejected_cleanly(tmp_path, payload):
    path = tmp_path / "bad.pdf"
    path.write_bytes(payload)
    with pytest.raises(PdfDamagedError, match="bad.pdf appears damaged or unsupported"):
        open_pdf(path)


def test_messages_name_the_file_never_the_path(tmp_path):
    path = tmp_path / "secret-folder" / "x.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF-1.4\nbroken")
    with pytest.raises(PdfError) as info:
        open_pdf(path, "Report.pdf")
    assert "Report.pdf" in str(info.value) and "secret-folder" not in str(info.value)


def test_password_protected_pdf_needs_a_password(tmp_path):
    doc = Document()
    doc.new_page().text(72, 720, "classified", 12)
    path = doc.save(tmp_path / "locked.pdf", user_password="user")
    with pytest.raises(PdfPasswordError, match="password-protected"):
        open_pdf(path)


def test_damaged_file_after_an_encrypted_one_is_not_called_encrypted(tmp_path):
    """PDFium's last error is global and not set by every failure."""
    doc = Document()
    doc.new_page().text(72, 720, "classified", 12)
    locked = doc.save(tmp_path / "locked.pdf", user_password="user")
    with pytest.raises(PdfPasswordError):
        open_pdf(locked)
    test_document_without_pages_is_rejected(tmp_path)


def test_document_without_pages_is_rejected(tmp_path):
    path = tmp_path / "empty.pdf"
    body = b"%PDF-1.4\n"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [] /Count 0 >>",
    ]
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(body)
    body += b"xref\n0 3\n0000000000 65535 f \n"
    body += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    body += f"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(body)
    with pytest.raises((PdfNoPagesError, PdfDamagedError)) as info:
        open_pdf(path)
    assert "password" not in str(info.value)


def test_engine_versions_are_reported():
    info = engine_versions()
    assert info["engine"] == "PDFium" and info["pypdfium2"] and info["pdfium"]


# --- rendering ------------------------------------------------------------------------


def test_render_is_a_valid_png_of_the_requested_scale(tmp_path):
    with open_pdf(make_pdf(tmp_path / "a.pdf", ["hello"])) as doc:
        for scale in (1.0, 1.5, 2.0):
            page = doc.render(0, scale)
            w, h, channels, _ = _decode_png(page.png)
            assert (w, h) == (page.width, page.height)
            assert abs(w - 595 * scale) <= 1 and abs(h - 842 * scale) <= 1
            assert channels == 3


def test_render_refuses_absurd_pixel_counts(tmp_path):
    doc = Document()
    doc.new_page(width=14400, height=14400).text(72, 14300, "Huge", 12)
    with open_pdf(doc.save(tmp_path / "huge.pdf")) as pdf:
        with pytest.raises(PdfError, match="too large"):
            pdf.render(0, 1.0)  # 207 MP; callers cap the long side first


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_rendering_orientation_follows_rotate(tmp_path, rotate):
    """A black square drawn at the unrotated page's top-left corner lands
    where /Rotate puts that corner."""
    doc = Document()
    doc.new_page(rotate=rotate).rect(0, 742, 100, 842)
    with open_pdf(doc.save(tmp_path / "r.pdf")) as pdf:
        size = pdf.page_size(0)
        assert size.rotation == rotate
        assert (size.width, size.height) == (
            (595, 842) if rotate in (0, 180) else (842, 595)
        )
        ink = _ink_box(pdf.render(0, 0.5).png)
    corners = [_display(rotate, x, y, 595, 842) for x, y in ((0, 742), (100, 842))]
    expected = (
        min(c[0] for c in corners),
        min(c[1] for c in corners),
        max(c[0] for c in corners),
        max(c[1] for c in corners),
    )
    assert all(abs(a - b) < 0.01 for a, b in zip(ink, expected, strict=True))


def test_crop_box_is_what_is_displayed(tmp_path):
    doc = Document()
    page = doc.new_page(crop=(100, 200, 400, 600))
    page.rect(100, 500, 150, 600)  # top-left corner of the crop box
    page.rect(0, 0, 50, 50)  # outside the crop box: never shown
    with open_pdf(doc.save(tmp_path / "c.pdf")) as pdf:
        size = pdf.page_size(0)
        assert (size.width, size.height) == (300, 400)
        ink = _ink_box(pdf.render(0, 1.0).png)
    assert ink[0] < 0.01 and ink[1] < 0.01
    assert abs(ink[2] - 50 / 300) < 0.01 and abs(ink[3] - 100 / 400) < 0.01


def test_annotation_appearances_are_drawn(tmp_path):
    doc = Document()
    page = doc.new_page()
    ap = doc.add_object(
        b"<< /Type /XObject /Subtype /Form /BBox [0 0 100 100] /Length 18 >>\n"
        b"stream\n0 0 100 100 re f\nendstream"
    )
    page.annots.append(
        "<< /Type /Annot /Subtype /Square /Rect [200 300 300 400]"
        f" /AP << /N {{ref:{ap}}} >> >>"
    )
    with open_pdf(doc.save(tmp_path / "an.pdf")) as pdf:
        ink = _ink_box(pdf.render(0, 0.5).png)
    assert abs(ink[0] - 200 / 595) < 0.01 and abs(ink[1] - (1 - 400 / 842)) < 0.01


def test_png_writer_round_trips_gray_and_rgb():
    rgb = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9])
    w, h, c, px = _decode_png(encode_png(rgb, 2, 2, 6, 3))
    assert (w, h, c, px) == (2, 2, 3, rgb)
    padded = bytes([0, 128, 99, 255, 7, 99])  # stride 3: one padding byte per row
    assert _decode_png(encode_png(padded, 2, 2, 3, 1))[3] == bytes([0, 128, 255, 7])
    with pytest.raises(ValueError):
        encode_png(b"", 1, 1, 4, 4)


# --- text and words -------------------------------------------------------------------


def _words(tmp_path: Path, build, name: str = "w.pdf"):  # type: ignore[no-untyped-def]
    doc = Document()
    build(doc.new_page())
    with open_pdf(doc.save(tmp_path / name)) as pdf:
        return pdf.page_text(0), pdf.page_words(0)


def test_words_split_on_spaces_and_keep_punctuation(tmp_path):
    text, words = _words(
        tmp_path, lambda p: p.text(72, 700, "Hello, world! (Quoted) 3.14 e-mail.", 12)
    )
    assert [w.text for w in words] == [
        "Hello,",
        "world!",
        "(Quoted)",
        "3.14",
        "e-mail.",
    ]
    assert text == "Hello, world! (Quoted) 3.14 e-mail."
    assert all(0 <= w.x0 < w.x1 <= 1 and 0 <= w.y0 < w.y1 <= 1 for w in words)
    assert [w.x0 for w in words] == sorted(w.x0 for w in words)


def test_lines_are_numbered_and_never_merged(tmp_path):
    text, words = _words(
        tmp_path,
        lambda p: p.lines(72, 760, ["alpha beta", "gamma delta", "epsilon"], 12),
    )
    assert text == "alpha beta\ngamma delta\nepsilon"
    lines = {}
    for w in words:
        lines.setdefault(w.line, []).append(w.text)
    assert list(lines.values()) == [["alpha", "beta"], ["gamma", "delta"], ["epsilon"]]


def test_line_end_hyphen_survives_for_the_normaliser(tmp_path):
    from litenotebook.ingest.pdf import normalize_page_text

    text, words = _words(
        tmp_path,
        lambda p: p.lines(72, 760, ["a well-", "known hyphen-", "ated word"], 12),
    )
    assert "hyphen-\nated" in text
    # the normaliser re-joins any line-end hyphen before a lower-case letter
    # (unchanged behaviour), so "well-known" also becomes "wellknown"
    assert normalize_page_text(text) == "a wellknown hyphenated word"
    assert "hyphen-" in [w.text for w in words]


def test_unicode_is_extracted_exactly(tmp_path):
    sample = "Ångström café naïve Straße 東京 Ελληνικά ∑ x² ≤ y “q” — ½"
    text, words = _words(tmp_path, lambda p: p.text(40, 700, sample, 11))
    assert text == sample
    assert " ".join(w.text for w in words) == sample


def test_repeated_words_get_distinct_boxes(tmp_path):
    _, words = _words(tmp_path, lambda p: p.text(72, 700, "echo echo echo", 12))
    assert [w.text for w in words] == ["echo"] * 3
    assert words[0].x1 <= words[1].x0 + 1e-6 and words[1].x1 <= words[2].x0 + 1e-6


def test_tj_spacing_without_space_characters_still_separates_words(tmp_path):
    """TeX-style: words positioned with TJ offsets, no space glyphs; kerning
    inside a word (small offsets) never splits it."""
    text, words = _words(
        tmp_path,
        lambda p: p.raw(
            "BT /F0 12 Tf 72 700 Td [(W) 80 (ave) -333 (functions) -333 (are) "
            "-2000 (justified)] TJ ET"
        ),
    )
    assert [w.text for w in words] == ["Wave", "functions", "are", "justified"]
    assert len({w.line for w in words}) == 1  # justified gaps never split a line


def test_far_apart_text_objects_on_one_baseline_are_separate_lines(tmp_path):
    """Table cells: MuPDF's rule (a gap over 0.8 em between objects)."""
    text, words = _words(
        tmp_path,
        lambda p: p.text(72, 700, "1", 10).text(100, 700, "North arrow", 10),
    )
    assert text == "1\nNorth arrow"
    assert words[0].line != words[1].line


def test_each_glyph_its_own_object_and_space_objects_dropped(tmp_path):
    """Some producers draw every glyph (and every space) as its own text
    object; the space objects yield no characters in PDFium."""
    ops = []
    x = 72.0
    for ch in "Single Words":
        ops.append(f"BT /F0 32 Tf {x:.2f} 700 Td ({ch}) Tj ET")
        x += text_width(ch, 32) * (0.55 if ch == " " else 1.0)  # a narrow space
    text, words = _words(tmp_path, lambda p: p.raw("\n".join(ops)))
    assert [w.text for w in words] == ["Single", "Words"]
    assert text == "Single Words"


def test_content_order_is_kept_on_a_rotated_page(tmp_path):
    """PDFium's own text order mixes lines on /Rotate 90 pages; the engine
    keeps the order the document draws its text."""
    lines = [f"line {n} of the sideways page" for n in range(1, 9)]
    doc = Document()
    doc.new_page(rotate=90).lines(72, 780, lines, 11)
    with open_pdf(doc.save(tmp_path / "r.pdf")) as pdf:
        assert pdf.page_text(0) == "\n".join(lines)


def test_vertical_and_tilted_text_form_words(tmp_path):
    text, words = _words(
        tmp_path,
        lambda p: p.text(100, 300, "Vertical label", 14, angle=90).text(
            300, 400, "Tilted words", 12, angle=30
        ),
    )
    assert text == "Vertical label\nTilted words"
    vertical = words[:2]
    assert [w.text for w in vertical] == ["Vertical", "label"]
    assert vertical[0].y0 > vertical[1].y1 - 1e-6  # reads bottom-up on screen


def test_scanned_page_has_no_words(tmp_path):
    path = make_pdf(tmp_path / "scan.pdf", ["A scanned page of words"], image_only={1})
    with open_pdf(path) as pdf:
        assert pdf.page_words(0) == [] and pdf.page_text(0) == ""
        assert pdf.render(0, 1.0).width == 595


# --- the coordinate convention ------------------------------------------------------


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_word_boxes_follow_the_displayed_page(tmp_path, rotate):
    """Word boxes are normalised to the displayed page (rotation and crop box
    applied, top-left origin) and cover the rendered glyphs."""
    doc = Document()
    doc.new_page(rotate=rotate).text(72, 700, "Marker", 40)
    with open_pdf(doc.save(tmp_path / "m.pdf")) as pdf:
        (word,) = pdf.page_words(0)
        ink = _ink_box(pdf.render(0, 1.0).png)
    width = text_width("Marker", 40)
    corners = [
        _display(rotate, x, y, 595, 842) for x, y in ((72, 700), (72 + width, 728))
    ]
    cx = sum(c[0] for c in corners) / 2
    cy = sum(c[1] for c in corners) / 2
    assert abs((word.x0 + word.x1) / 2 - cx) < 0.02
    assert abs((word.y0 + word.y1) / 2 - cy) < 0.03
    slack = 0.01  # the box is a line box: it contains the ink
    assert word.x0 - slack <= ink[0] and ink[2] <= word.x1 + slack
    assert word.y0 - slack <= ink[1] and ink[3] <= word.y1 + slack


def test_word_boxes_with_crop_box_and_rotation(tmp_path):
    doc = Document()
    doc.new_page(rotate=90, crop=(50, 100, 545, 742)).text(100, 650, "Cropped", 30)
    with open_pdf(doc.save(tmp_path / "cr.pdf")) as pdf:
        (word,) = pdf.page_words(0)
        ink = _ink_box(pdf.render(0, 1.0).png)
    assert word.x0 - 0.01 <= ink[0] and ink[2] <= word.x1 + 0.01
    assert word.y0 - 0.01 <= ink[1] and ink[3] <= word.y1 + 0.01


# --- highlights -----------------------------------------------------------------------


PARA = (
    "The harbour of Velt is guarded by the Orrin mountains. Tidal locks open "
    "twice a day and the ferry to Glacier Bay leaves at dawn. "
)


def _geometry(path: Path, page: int = 0) -> PageGeometry:
    with open_pdf(path) as pdf:
        words = [
            Word(w.text, w.x0, w.y0, w.x1, w.y1, w.block, w.line)
            for w in pdf.page_words(page)
        ]
    return PageGeometry(page + 1, 1, 1, words)


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_citation_passages_highlight_at_start_middle_end(tmp_path, rotate):
    doc = Document()
    doc.new_page(rotate=rotate).textbox(
        (40, 40, 555, 802), PARA * 4 + "Final words.", 11
    )
    path = doc.save(tmp_path / "h.pdf")
    geometry = _geometry(path)
    for phrase, hint in (
        ("The harbour of Velt is guarded", 0.0),
        ("ferry to Glacier Bay leaves at dawn. The harbour", 0.5),
        ("leaves at dawn. Final words.", 1.0),
    ):
        rects = match_text(geometry, phrase, hint)
        assert rects, phrase
        assert all(0 <= v <= 1 for r in rects for v in r)


def test_multi_line_and_hyphenated_passages_highlight_every_line(tmp_path):
    doc = Document()
    doc.new_page().lines(
        72, 760, ["the tidal locks open twice a", "day; a well-", "known ferry"], 12
    )
    geometry = _geometry(doc.save(tmp_path / "m.pdf"))
    rects = match_text(geometry, "twice a day; a wellknown ferry", None)
    assert rects and len(rects) == 3


def test_repeated_phrase_uses_the_position_hint(tmp_path):
    doc = Document()
    doc.new_page().lines(
        72, 760, ["the same phrase", "filler text here", "the same phrase"], 12
    )
    geometry = _geometry(doc.save(tmp_path / "r.pdf"))
    first = match_text(geometry, "the same phrase", 0.0)
    last = match_text(geometry, "the same phrase", 0.99)
    assert first and last and first[0][1] < last[0][1]


def test_text_that_is_not_on_the_page_is_never_highlighted(tmp_path):
    doc = Document()
    doc.new_page().text(72, 700, "Only this sentence exists here.", 12)
    geometry = _geometry(doc.save(tmp_path / "n.pdf"))
    assert match_text(geometry, "a completely different passage", None) is None


# --- PyMuPDF is gone ------------------------------------------------------------------


def test_no_production_code_imports_pymupdf():
    offenders = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "src").rglob("*.py")
        if "import fitz" in p.read_text("utf-8")
        or "import pymupdf" in p.read_text("utf-8")
        or "from pymupdf" in p.read_text("utf-8")
    ]
    assert offenders == []


def test_pymupdf_is_not_a_dependency():
    text = (ROOT / "pyproject.toml").read_text("utf-8").lower()
    assert "pymupdf" not in text and "pypdfium2" in text
    lock = (ROOT / "uv.lock").read_text("utf-8").lower()
    assert 'name = "pymupdf"' not in lock and 'name = "pypdfium2"' in lock


def test_pdf_workflows_run_with_pymupdf_blocked(tmp_path):
    """Upload validation, extraction, OCR rendering, Reader pages and word
    geometry, in a fresh interpreter where ``import fitz`` and ``import
    pymupdf`` fail."""
    pdf = make_pdf(tmp_path / "p.pdf", ["Velt harbour text page", ""], image_only={2})
    code = f"""
import sys
sys.modules["fitz"] = None
sys.modules["pymupdf"] = None
for name in ("fitz", "pymupdf"):
    try:
        __import__(name)
    except ImportError:
        pass
    else:
        raise SystemExit(name + " importable")
from pathlib import Path
from litenotebook.ingest.pdf import inspect_pdf, extract_pages
from litenotebook.reader.geometry import page_geometry
from litenotebook.pdf import open_pdf
p = Path({str(pdf)!r})
assert inspect_pdf(p) == 2
pages = extract_pages(p)
assert pages[0].method == "text" and "Velt" in pages[0].text
assert page_geometry(p, 1).words[0].text == "Velt"
with open_pdf(p) as doc:
    assert doc.render(1, 1.0).png.startswith(b"\\x89PNG")
print("OK", "pymupdf" in sys.modules and sys.modules["pymupdf"] is not None)
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "OK False"
