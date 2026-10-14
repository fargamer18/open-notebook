"""The PDFium implementation of LiteNotebook's PDF boundary (pypdfium2).

Only parsing, text extraction and rasterisation are used. Forms are never
initialised and no JavaScript engine is present in the pypdfium2 wheels, so
document actions, scripts, embedded files and links are never executed or
followed. Every call holds ``PDF_LOCK`` (see ``engine.py``).

Text and word reconstruction
----------------------------
PDFium exposes characters, not words, and its own text order is built with
display-space heuristics that mix lines on some rotated pages. LiteNotebook
therefore uses PDFium only for each character's code, box, origin, angle and
text object, and rebuilds the rest itself (``_layout``), deterministically:

* Characters are ordered by the content-stream position of their text object
  (Form XObjects expanded in place) and then by PDFium's index: the order the
  document draws its text, which is what the previous engine used.
  PDFium's generated characters (spaces and line breaks it guessed) are
  ignored; spaces and breaks are decided here.
* Positions are measured in each character's own text frame (its angle), so
  rotated pages and rotated text are grouped like ordinary horizontal text.
* A new line starts when the baseline moves by more than half a line height,
  the text runs backwards, the direction turns, or the next text object
  starts more than 0.8 em after the previous glyph or space (table cells,
  columns). Gaps inside one text object (justified ``TJ`` spacing) never
  split a line.
* A word ends at a space or tab character (other spaces such as U+00A0 and
  U+200B stay part of the text, as before), a new line, a gap from the
  previous glyph to the next glyph's origin wider than 0.15 em (MuPDF's
  word-space threshold), or where PDFium itself generated a space between
  two glyphs that are neighbours in its order (it knows the font's space
  width, which matters for fonts with narrow spaces), or where a text object
  that yields no characters is drawn in between (producers that draw each
  space as its own object, which PDFium leaves out of the text page).
  Text at angles that are not a multiple of 90 degrees only splits at real
  spaces, because its axis-aligned boxes cannot measure gaps.
* PDFium's end-of-line hyphen marker is turned back into ``-``; page text
  keeps ``-\n`` so the ingestion normaliser re-joins hyphenated words as
  before.

Boxes are PDFium's "loose" character boxes (font ascent to descent, like a
text line), joined per word in PDF user space and then mapped to the
displayed page with the same transform PDFium uses for rendering
(``FPDF_PageToDevice``): crop box, ``/Rotate`` and the top-left origin are
handled in one place.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from litenotebook.pdf.engine import PDF_LOCK
from litenotebook.pdf.models import (
    PageSize,
    PdfDamagedError,
    PdfError,
    PdfNoPagesError,
    PdfPasswordError,
    RenderedPage,
    WordBox,
)
from litenotebook.pdf.png import encode_png

MAX_RENDER_PIXELS = 40_000_000  # defence in depth; callers cap the long side
_HYPHEN_MARKS = {0x02, 0xFFFE}  # PDFium: a hyphen at a line end
SPACE_GAP_EM = 0.15  # a gap this wide (in em) separates words
LINE_GAP_EM = 0.8  # a wider gap between text objects starts a new line
_BREAK_SPACES = {0x20, 0x09, 0x3000}  # word breaks; other spaces are text
_DEVICE = 1_000_000  # device size used to read PDFium's page transform


def _damaged(name: str) -> str:
    return (
        f"{name} appears damaged or unsupported. Re-save or export it as a PDF "
        "again (for example with Print to PDF) and upload the new file."
    )


def _reset_last_error() -> None:
    """PDFium's last error is process-wide and not set by every failure, so a
    stale "incorrect password" from an earlier file could label a damaged one.
    A load that is certain to fail sets it to a known, non-password value.
    Caller holds ``PDF_LOCK``."""
    junk = ctypes.create_string_buffer(b"not a pdf")
    doc = pdfium_c.FPDF_LoadMemDocument(junk, len(junk.raw), None)
    if doc:  # pragma: no cover - never a valid document
        pdfium_c.FPDF_CloseDocument(doc)


class PdfiumDocument:
    def __init__(self, path: Path, name: str) -> None:
        self.name = name
        self._doc: Any = None
        with PDF_LOCK:
            _reset_last_error()
            try:
                self._doc = pdfium.PdfDocument(str(path))
            except pdfium.PdfiumError as e:
                code = getattr(e, "err_code", None)
                if code == pdfium_c.FPDF_ERR_PASSWORD:
                    raise PdfPasswordError(f"{name} is password-protected") from None
                if code == pdfium_c.FPDF_ERR_SECURITY:
                    raise PdfPasswordError(
                        f"{name} uses an encryption method LiteNotebook cannot open"
                    ) from None
                raise PdfDamagedError(_damaged(name)) from None
            except OSError as e:
                raise PdfError(f"{name} could not be read") from e
            try:
                count = len(self._doc)
            except Exception:
                self._close_locked()
                raise PdfDamagedError(_damaged(name)) from None
            if count < 1:
                self._close_locked()
                raise PdfNoPagesError(f"{name} has no pages")
            self._count = int(count)

    # --- lifetime ---------------------------------------------------------------

    def _close_locked(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def close(self) -> None:
        with PDF_LOCK:
            self._close_locked()

    def __enter__(self) -> PdfiumDocument:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def page_count(self) -> int:
        return self._count

    @contextmanager
    def _page(self, index: int) -> Iterator[Any]:
        """A loaded page, closed afterwards. Caller holds ``PDF_LOCK``."""
        if self._doc is None:
            raise PdfError(f"{self.name} is closed")
        if not 0 <= index < self._count:
            raise IndexError(f"page index {index} out of range")
        page = self._doc[index]
        try:
            yield page
        finally:
            page.close()

    # --- pages ------------------------------------------------------------------

    def page_size(self, index: int) -> PageSize:
        with PDF_LOCK, self._page(index) as page:
            width, height = page.get_size()
            return PageSize(float(width), float(height), int(page.get_rotation()))

    def page_text(self, index: int) -> str:
        """Page text in drawing order: lines end with ``\\n``, words are
        separated by one space, an end-of-line hyphen stays ``-\\n``."""
        return "\n".join(
            " ".join(word.text for word in line) for line in self._lines(index)
        )

    def page_words(self, index: int) -> list[WordBox]:
        out: list[WordBox] = []
        lines = self._lines(index)
        if not lines:
            return out
        with PDF_LOCK, self._page(index) as page:
            transform = _page_transform(page)
        for line in lines:
            for word in line:
                box = word.normalised(transform)
                if box is not None:
                    out.append(WordBox(word.text, *box, word.block, word.line))
        return out

    def _lines(self, index: int) -> list[list[_Word]]:
        with PDF_LOCK, self._page(index) as page:
            textpage = page.get_textpage()
            try:
                chars = _chars(page, textpage)
            finally:
                textpage.close()
        return _layout(chars)

    def render(self, index: int, scale: float) -> RenderedPage:
        """The displayed page (crop box, rotation applied) as a PNG."""
        with PDF_LOCK, self._page(index) as page:
            width, height = page.get_size()
            if width * scale * height * scale > MAX_RENDER_PIXELS:
                raise PdfError(f"{self.name}: page {index + 1} is too large to render")
            bitmap = page.render(
                scale=scale,
                may_draw_forms=False,  # forms are never initialised
                draw_annots=True,  # annotation appearances, as before
                rev_byteorder=True,  # RGB
                fill_color=(255, 255, 255, 255),
            )
            try:
                w, h, stride = int(bitmap.width), int(bitmap.height), int(bitmap.stride)
                channels = int(bitmap.n_channels)
                pixels = bytes(bitmap.buffer)
            finally:
                bitmap.close()
        return RenderedPage(encode_png(pixels, w, h, stride, channels), w, h)


# --- geometry helpers ------------------------------------------------------------


def _page_transform(page: Any) -> tuple[float, float, float, float, float, float]:
    """PDF user space → normalised displayed page, as an affine map
    (a, b, c, d, e, f): nx = a*x + c*y + e, ny = b*x + d*y + f."""
    width, height = page.get_size()
    sx = _DEVICE
    sy = max(1, round(_DEVICE * float(height) / float(width or 1)))
    dx, dy = ctypes.c_int(), ctypes.c_int()

    def device(x: float, y: float) -> tuple[float, float]:
        pdfium_c.FPDF_PageToDevice(
            page.raw,
            0,
            0,
            sx,
            sy,
            0,
            ctypes.c_double(x),
            ctypes.c_double(y),
            ctypes.byref(dx),
            ctypes.byref(dy),
        )
        return dx.value / sx, dy.value / sy

    k = 1000.0
    x0, y0 = device(0.0, 0.0)
    x1, y1 = device(k, 0.0)
    x2, y2 = device(0.0, k)
    return ((x1 - x0) / k, (y1 - y0) / k, (x2 - x0) / k, (y2 - y0) / k, x0, y0)


def _clamp(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


@dataclass(frozen=True)
class _Char:
    text: str  # "" for whitespace (a word break)
    box: tuple[float, float, float, float]  # left, bottom, right, top (user)
    angle: float  # counter-clockwise, radians, 0..2pi
    aligned: bool  # the angle is a multiple of 90 degrees
    u0: float  # extent along the text direction
    u1: float
    v: float  # baseline position across the text direction
    origin: float  # glyph origin along the text direction
    height: float  # line height (loose box across the text direction)
    em: float  # font size in user units
    rank: int  # drawing position of the character's text object
    index: int  # PDFium character index
    after: int  # index of the previous real character in PDFium's order
    spaced: bool  # PDFium generated a space between ``after`` and this one


@dataclass
class _Word:
    chars: list[_Char]
    block: int
    line: int

    @property
    def text(self) -> str:
        return "".join(c.text for c in self.chars)

    def normalised(
        self, t: tuple[float, float, float, float, float, float]
    ) -> tuple[float, float, float, float] | None:
        left = min(c.box[0] for c in self.chars)
        bottom = min(c.box[1] for c in self.chars)
        right = max(c.box[2] for c in self.chars)
        top = max(c.box[3] for c in self.chars)
        a, b, c, d, e, f = t
        xs, ys = [], []
        for x, y in ((left, bottom), (left, top), (right, bottom), (right, top)):
            xs.append(a * x + c * y + e)
            ys.append(b * x + d * y + f)
        nx0, nx1 = _clamp(min(xs)), _clamp(max(xs))
        ny0, ny1 = _clamp(min(ys)), _clamp(max(ys))
        if nx1 <= nx0 or ny1 <= ny0:
            return None
        return nx0, ny0, nx1, ny1


def _void_variant(fn: Any, argtypes: list[Any]) -> Any:
    """The same PDFium function returning/taking plain addresses (ints): the
    per-character loop would otherwise spend most of its time converting
    ctypes pointer objects. Keeps the library's calling convention."""
    copy = type(fn)(ctypes.cast(fn, ctypes.c_void_p).value)
    copy.argtypes = argtypes
    copy.restype = fn.restype if fn.restype in (ctypes.c_int, None) else ctypes.c_void_p
    return copy


_vp, _int = ctypes.c_void_p, ctypes.c_int
_TEXT_OBJECT = _void_variant(pdfium_c.FPDFText_GetTextObject, [_vp, _int])
_PAGE_OBJECT = _void_variant(pdfium_c.FPDFPage_GetObject, [_vp, _int])
_FORM_COUNT = _void_variant(pdfium_c.FPDFFormObj_CountObjects, [_vp])
_FORM_OBJECT = _void_variant(pdfium_c.FPDFFormObj_GetObject, [_vp, ctypes.c_ulong])
_OBJECT_TYPE = _void_variant(pdfium_c.FPDFPageObj_GetType, [_vp])


def _object_order(page: Any) -> dict[int, int]:
    """Text object address → drawing position, Form XObjects expanded."""
    order: dict[int, int] = {}

    def walk(count: int, get: Any) -> None:
        for i in range(count):
            obj = get(i)
            if not obj:
                continue
            kind = _OBJECT_TYPE(obj)
            if kind == pdfium_c.FPDF_PAGEOBJ_TEXT:
                order.setdefault(obj, len(order))
            elif kind == pdfium_c.FPDF_PAGEOBJ_FORM and len(order) < 200_000:
                walk(int(_FORM_COUNT(obj)), lambda j, o=obj: _FORM_OBJECT(o, j))

    raw = page.raw
    walk(pdfium_c.FPDFPage_CountObjects(raw), lambda i: _PAGE_OBJECT(raw, i))
    return order


def _codes(raw: Any, count: int) -> list[int]:
    """Every character's code point in one call when PDFium's text has one
    UTF-16 unit per character (always, except for characters beyond the BMP)."""
    buffer = (ctypes.c_ushort * (count + 1))()
    written = pdfium_c.FPDFText_GetText(raw, 0, count, buffer)
    if written == count + 1:
        units = list(buffer[:count])
        if not any(0xD800 <= u <= 0xDFFF for u in units):
            return units
    return [pdfium_c.FPDFText_GetUnicode(raw, i) for i in range(count)]


@dataclass(frozen=True)
class _Frame:
    """Per text object: direction and size (shared by all its glyphs)."""

    angle: float
    ux: float
    uy: float
    aligned: bool
    em: float


def _frame(raw: Any, i: int) -> _Frame:
    angle = pdfium_c.FPDFText_GetCharAngle(raw, i)  # clockwise, or -1
    angle = (2 * math.pi - angle) % (2 * math.pi) if angle >= 0 else 0.0
    em = float(pdfium_c.FPDFText_GetFontSize(raw, i))
    return _Frame(angle, math.cos(angle), math.sin(angle), _axis_aligned(angle), em)


def _chars(page: Any, textpage: Any) -> list[_Char]:
    """Real characters in drawing order (caller holds ``PDF_LOCK``)."""
    raw = textpage.raw
    count = textpage.count_chars()
    if count <= 0:
        return []
    order = _object_order(page)
    codes = _codes(raw, count)
    rect = pdfium_c.FS_RECTF()
    rect_ref = ctypes.byref(rect)
    ox, oy = ctypes.c_double(), ctypes.c_double()
    ox_ref, oy_ref = ctypes.byref(ox), ctypes.byref(oy)
    frames: dict[int, _Frame] = {}
    is_generated = pdfium_c.FPDFText_IsGenerated
    loose_box = pdfium_c.FPDFText_GetLooseCharBox
    char_origin = pdfium_c.FPDFText_GetCharOrigin
    unknown = len(order)
    keyed: list[tuple[int, int, _Char]] = []
    last_real, spaced = -1, False
    for i in range(count):
        code = codes[i]
        if is_generated(raw, i) == 1:
            spaced = spaced or code == 0x20
            continue
        if code in _HYPHEN_MARKS:
            ch = "-"
        elif code in _BREAK_SPACES:
            ch = ""
        elif code < 32 or 0xD800 <= code <= 0xDFFF:
            continue  # line breaks are decided here; surrogates are malformed
        else:
            ch = chr(code)
        if not loose_box(raw, i, rect_ref):
            continue
        left, right = min(rect.left, rect.right), max(rect.left, rect.right)
        bottom, top = min(rect.bottom, rect.top), max(rect.bottom, rect.top)
        if right - left <= 0 and ch:
            continue
        if not char_origin(raw, i, ox_ref, oy_ref):
            ox.value, oy.value = left, bottom
        obj = _TEXT_OBJECT(raw, i) or 0
        frame = frames.get(obj) if obj else None
        if frame is None:
            frame = _frame(raw, i)
            if obj:
                frames[obj] = frame
        ux, uy = frame.ux, frame.uy
        x, y = ox.value, oy.value
        if frame.aligned:
            ua, ub = left * ux + bottom * uy, right * ux + top * uy
            va, vb = -left * uy + bottom * ux, -right * uy + top * ux
            u0, u1 = (ua, ub) if ua <= ub else (ub, ua)
            height = abs(vb - va)
        else:
            u0 = u1 = x * ux + y * uy
            height = max(right - left, top - bottom)
        rank = order.get(obj, unknown) if obj else unknown
        char = _Char(
            ch,
            (left, bottom, right, top),
            frame.angle,
            frame.aligned,
            u0,
            u1,
            -x * uy + y * ux,
            x * ux + y * uy,
            height,
            frame.em or height or 1.0,
            rank,
            i,
            last_real,
            spaced,
        )
        keyed.append((rank, i, char))
        last_real, spaced = i, False
    keyed.sort(key=lambda k: (k[0], k[1]))
    return [c for _, _, c in keyed]


def _silent_between(chars: list[_Char]) -> set[tuple[int, int]]:
    """(rank, rank) pairs of consecutive characters with a text object drawn
    between them that produced no characters."""
    ranks = {c.rank for c in chars}
    pairs = set()
    for a, b in zip(chars, chars[1:], strict=False):
        if b.rank - a.rank > 1 and any(
            r not in ranks for r in range(a.rank + 1, b.rank)
        ):
            pairs.add((a.rank, b.rank))
    return pairs


def _axis_aligned(angle: float) -> bool:
    quarter = angle / (math.pi / 2)
    return abs(quarter - round(quarter)) < 0.03


def _turned(a: float, b: float) -> bool:
    diff = abs(a - b) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff) > 0.3


def _layout(chars: list[_Char]) -> list[list[_Word]]:
    """Group characters into lines of words (see the module docstring)."""
    lines: list[list[_Word]] = []
    word: list[_Char] = []
    prev: _Char | None = None
    block = line_no = 0
    broken = False  # a whitespace character since the last glyph
    silent = _silent_between(chars)

    def end_word() -> None:
        nonlocal word
        if word:
            if not lines:
                lines.append([])
            lines[-1].append(_Word(word, block, line_no))
            word = []

    for c in chars:
        if prev is not None and (
            c.angle == prev.angle or not _turned(c.angle, prev.angle)
        ):
            line_h = max(c.height, prev.height) or 1.0
            same_line = abs(c.v - prev.v) <= 0.5 * line_h and c.u1 >= prev.u0
        else:
            same_line = prev is None
        if prev is not None and same_line:
            em = max(c.em, prev.em)
            gap = c.u0 - prev.u1
            if gap > LINE_GAP_EM * em and c.aligned and c.rank != prev.rank:
                same_line = False
        if prev is not None and not same_line:
            end_word()
            far = prev is not None and (
                _turned(c.angle, prev.angle)
                or abs(c.v - prev.v) > 1.6 * max(c.height, prev.height, 1e-6)
            )
            block += 1 if far else 0
            line_no += 1
            lines.append([])
            broken = False
        if not c.text:  # whitespace: ends the word, keeps the position
            end_word()
            broken = True
            if prev is None or not same_line or c.u1 > prev.u1:
                prev = c
            continue
        if prev is not None and same_line and word:
            gap = min(c.u0, c.origin) - prev.u1
            if (
                broken
                or (c.spaced and c.after == prev.index)
                or (prev.rank, c.rank) in silent
                or (c.aligned and gap > SPACE_GAP_EM * max(c.em, prev.em))
            ):
                end_word()
        elif broken:
            end_word()
        broken = False
        word.append(c)
        prev = c
    end_word()
    return [line for line in lines if line]
