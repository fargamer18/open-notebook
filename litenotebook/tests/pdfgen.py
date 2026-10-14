"""A small, deterministic PDF writer for test fixtures (no PDF library).

LiteNotebook's runtime only *reads* PDFs. Tests need to *write* them —
ordinary text, rotated pages, crop boxes, Unicode, images, annotations,
actions, encryption — so this module produces them directly. Output depends
only on the input: no dates, no random IDs.

Text uses the standard Helvetica font with explicit ``/Widths`` (the Adobe
AFM metrics), so layout here and glyph boxes in any reader agree. Characters
outside WinAnsi get codes in extra fonts whose ``/ToUnicode`` CMap maps each
code back to the real character: extraction returns real Unicode even though
the drawn glyph is a placeholder.

Coordinates passed to ``Page.text`` are PDF user space (origin bottom-left,
points), exactly what goes into the content stream.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

_ASCII_WIDTHS = [
    278,
    278,
    355,
    556,
    556,
    889,
    667,
    191,
    333,
    333,
    389,
    584,
    278,
    333,
    278,
    278,
    556,
    556,
    556,
    556,
    556,
    556,
    556,
    556,
    556,
    556,
    278,
    278,
    584,
    584,
    584,
    556,
    1015,
    667,
    667,
    722,
    722,
    667,
    611,
    778,
    722,
    278,
    500,
    667,
    556,
    833,
    722,
    778,
    667,
    778,
    722,
    667,
    611,
    722,
    667,
    944,
    667,
    667,
    611,
    278,
    278,
    278,
    469,
    556,
    333,
    556,
    556,
    500,
    556,
    556,
    278,
    556,
    556,
    222,
    222,
    500,
    222,
    833,
    556,
    556,
    556,
    556,
    333,
    500,
    278,
    556,
    500,
    722,
    500,
    500,
    500,
    334,
    260,
    334,
    584,
]  # Helvetica, codes 32..126
_DEFAULT_WIDTH = 556
LINE_FACTOR = 1.2  # line height / font size for wrapped text


def _width_of_code(code: int) -> int:
    return _ASCII_WIDTHS[code - 32] if 32 <= code <= 126 else _DEFAULT_WIDTH


def text_width(text: str, size: float) -> float:
    total = 0
    for ch in text:
        try:
            code = ch.encode("cp1252")[0]
        except UnicodeEncodeError:
            code = 0
        total += _width_of_code(code)
    return total * size / 1000


def _escape(raw: bytes) -> bytes:
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _pdf_string(text: str) -> str:
    return (
        "(" + text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"
    )


@dataclass
class _Image:
    pixels: bytes  # RGB
    width: int
    height: int
    rect: tuple[float, float, float, float]  # x0, y0, x1, y1 user space


@dataclass
class Page:
    width: float = 595
    height: float = 842
    rotate: int = 0
    crop: tuple[float, float, float, float] | None = None
    media_origin: tuple[float, float] = (0, 0)
    ops: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    images: list[_Image] = field(default_factory=list)
    annots: list[str] = field(default_factory=list)  # raw dicts; may use {ref:N}
    extra_dict: str = ""  # raw keys added to the page dict (e.g. /AA)

    def text(
        self, x: float, y: float, s: str, size: float = 12, angle: float = 0
    ) -> Page:
        """Draw ``s`` with its baseline starting at (x, y), rotated by
        ``angle`` degrees counter-clockwise."""
        self.ops.append(("text", (x, y, s, size, angle)))
        return self

    def lines(
        self,
        x: float,
        top: float,
        lines: list[str],
        size: float = 12,
        leading: float | None = None,
    ) -> Page:
        """Lines of text; ``top`` is the top of the first line (user space)."""
        leading = leading or size * LINE_FACTOR
        for i, line in enumerate(lines):
            self.text(x, top - size * 0.8 - i * leading, line, size)
        return self

    def textbox(
        self, rect: tuple[float, float, float, float], text: str, size: float = 11
    ) -> Page:
        """Greedy word wrap into ``rect`` given top-left style (x0, top, x1,
        bottom) in *top-down* points, like PyMuPDF's ``insert_textbox``.
        Raises ``OverflowError`` when the text does not fit."""
        x0, top, x1, bottom = rect
        width = x1 - x0
        leading = size * LINE_FACTOR
        out: list[str] = []
        for para in text.split("\n"):
            line = ""
            for word in para.split(" "):
                candidate = f"{line} {word}" if line else word
                if line and text_width(candidate, size) > width:
                    out.append(line)
                    line = word
                else:
                    line = candidate
            out.append(line)
        if len(out) * leading > bottom - top + 0.01:
            raise OverflowError("text does not fit the box")
        user_top = self.height - top
        return self.lines(x0, user_top, out, size, leading)

    def raw(self, content: str) -> Page:
        """Content-stream operators as written (Helvetica is ``/F0``; text
        strings must be ASCII)."""
        self.ops.append(("raw", (content,)))
        return self

    def rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        rgb: tuple[float, float, float] = (0, 0, 0),
    ) -> Page:
        self.ops.append(("rect", (x0, y0, x1, y1, rgb)))
        return self

    def image(
        self,
        pixels: bytes,
        width: int,
        height: int,
        rect: tuple[float, float, float, float] | None = None,
    ) -> Page:
        self.images.append(
            _Image(pixels, width, height, rect or (0, 0, self.width, self.height))
        )
        return self


class Document:
    def __init__(self) -> None:
        self.pages: list[Page] = []
        self.catalog_extra = ""  # raw catalog keys, may use {ref:N}
        self.objects: list[bytes] = []  # extra objects referenced as {ref:N}
        self._fonts: list[dict[str, int]] = [{}]  # font index -> char -> code

    def new_page(self, width: float = 595, height: float = 842, **kw: object) -> Page:
        page = Page(width, height, **kw)  # type: ignore[arg-type]
        self.pages.append(page)
        return page

    def add_object(self, body: str | bytes) -> int:
        """An extra indirect object; returns N for use as ``{ref:N}``."""
        self.objects.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(self.objects) - 1

    # --- text encoding --------------------------------------------------------------

    def _code(self, ch: str) -> tuple[int, int]:
        try:
            code = ch.encode("cp1252")[0]
            if code >= 32:
                return 0, code
        except UnicodeEncodeError:
            pass
        for index, table in enumerate(self._fonts[1:], 1):
            if ch in table:
                return index, table[ch]
        table = self._fonts[-1] if len(self._fonts) > 1 else None
        if table is None or len(table) >= 200:
            table = {}
            self._fonts.append(table)
        table[ch] = 33 + len(table)
        return len(self._fonts) - 1, table[ch]

    def _runs(self, s: str) -> list[tuple[int, bytes]]:
        runs: list[tuple[int, bytearray]] = []
        for ch in s:
            font, code = self._code(ch)
            if runs and runs[-1][0] == font:
                runs[-1][1].append(code)
            else:
                runs.append((font, bytearray([code])))
        return [(f, bytes(b)) for f, b in runs]

    def _content(self, page: Page) -> bytes:
        out: list[bytes] = []
        for index, image in enumerate(page.images):
            x0, y0, x1, y1 = image.rect
            out.append(
                f"q {x1 - x0:.4f} 0 0 {y1 - y0:.4f} {x0:.4f} {y0:.4f} cm "
                f"/Im{index} Do Q\n".encode()
            )
        for kind, args in page.ops:
            if kind == "raw":
                out.append(str(args[0]).encode("ascii") + b"\n")
                continue
            if kind == "rect":
                x0, y0, x1, y1, (r, g, b) = args  # type: ignore[misc]
                out.append(
                    f"q {r} {g} {b} rg {x0} {y0} {x1 - x0} {y1 - y0} re f Q\n".encode()
                )
                continue
            x, y, s, size, angle = args  # type: ignore[misc]
            a = math.radians(float(angle))  # type: ignore[arg-type]
            c, sn = round(math.cos(a), 6), round(math.sin(a), 6)
            parts = [f"BT {c} {sn} {-sn} {c} {x:.3f} {y:.3f} Tm ".encode()]
            for font, raw in self._runs(str(s)):
                parts.append(f"/F{font} {size} Tf ".encode())
                parts.append(b"(" + _escape(raw) + b") Tj ")
            parts.append(b"ET\n")
            out.append(b"".join(parts))
        return b"".join(out)

    # --- serialisation --------------------------------------------------------------

    def save(
        self,
        path: Path,
        *,
        compress: bool = False,
        user_password: str | None = None,
        owner_password: str = "owner",
    ) -> Path:
        contents = [self._content(p) for p in self.pages]  # fills the font tables
        bodies: list[bytes | None] = []  # object n+1 at index n

        def reserve() -> int:
            bodies.append(None)
            return len(bodies)

        catalog, pages_ref = reserve(), reserve()
        font_refs = [reserve() for _ in self._fonts]
        extra_refs = [reserve() for _ in self.objects]

        def resolve(raw: str) -> str:
            for i, ref in enumerate(extra_refs):
                raw = raw.replace(f"{{ref:{i}}}", f"{ref} 0 R")
            return raw

        streams: dict[int, tuple[str, bytes]] = {}  # obj -> (dict keys, data)

        def stream(keys: str, data: bytes, deflate: bool) -> int:
            ref = reserve()
            if deflate:
                data = zlib.compress(data, 9)
                keys += " /Filter /FlateDecode"
            streams[ref] = (keys, data)
            return ref

        page_refs = []
        for page, content in zip(self.pages, contents, strict=True):
            ref = reserve()
            page_refs.append(ref)
            content_ref = stream("", content, compress)
            images = " ".join(
                f"/Im{i} {stream(self._image_keys(im), im.pixels, True)} 0 R"
                for i, im in enumerate(page.images)
            )
            annot_refs = []
            for annot in page.annots:
                aref = reserve()
                bodies[aref - 1] = resolve(annot).encode("latin-1")
                annot_refs.append(aref)
            fonts = " ".join(f"/F{i} {r} 0 R" for i, r in enumerate(font_refs))
            mx, my = page.media_origin
            media = f"[{mx} {my} {mx + page.width} {my + page.height}]"
            crop = (
                f" /CropBox [{' '.join(str(v) for v in page.crop)}]"
                if page.crop
                else ""
            )
            annots = (
                f" /Annots [{' '.join(f'{a} 0 R' for a in annot_refs)}]"
                if annot_refs
                else ""
            )
            body = (
                f"<< /Type /Page /Parent {pages_ref} 0 R /MediaBox {media}{crop}"
                f" /Rotate {page.rotate} /Contents {content_ref} 0 R"
                f" /Resources << /Font << {fonts} >> /XObject << {images} >> >>"
                f"{annots} {resolve(page.extra_dict)} >>"
            )
            bodies[ref - 1] = body.encode("latin-1")
        kids = " ".join(f"{r} 0 R" for r in page_refs)
        bodies[pages_ref - 1] = (
            f"<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>".encode()
        )
        bodies[catalog - 1] = (
            f"<< /Type /Catalog /Pages {pages_ref} 0 R {resolve(self.catalog_extra)} >>"
        ).encode("latin-1")
        for index, ref in enumerate(font_refs):
            bodies[ref - 1] = self._font(index, stream).encode()
        for raw, ref in zip(self.objects, extra_refs, strict=True):
            bodies[ref - 1] = resolve(raw.decode("latin-1")).encode("latin-1")

        file_id = hashlib.md5(b"".join(contents), usedforsecurity=False).digest()
        encrypt_ref = None
        key = b""
        if user_password is not None:
            o, u, key, p = _rc4_security(user_password, owner_password, file_id)
            encrypt_ref = reserve()
            bodies[encrypt_ref - 1] = (
                f"<< /Filter /Standard /V 2 /R 3 /Length 128 /P {p}"
                f" /O <{o.hex()}> /U <{u.hex()}> >>"
            ).encode()

        out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, body in enumerate(bodies, 1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n".encode()
            if number in streams:
                keys, data = streams[number]
                if key:
                    data = _rc4(_object_key(key, number), data)
                out += f"<< /Length {len(data)}{keys} >>\nstream\n".encode()
                out += data + b"\nendstream"
            else:
                assert body is not None, f"object {number} was never written"
                if key and number != encrypt_ref:
                    assert b"(" not in body, "encrypted fixtures: no dict strings"
                out += body
            out += b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(bodies) + 1}\n0000000000 65535 f \n".encode()
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode()
        enc = f" /Encrypt {encrypt_ref} 0 R" if encrypt_ref else ""
        out += (
            f"trailer\n<< /Size {len(bodies) + 1} /Root {catalog} 0 R{enc}"
            f" /ID [<{file_id.hex()}> <{file_id.hex()}>] >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
        Path(path).write_bytes(bytes(out))
        return Path(path)

    @staticmethod
    def _image_keys(image: _Image) -> str:
        return (
            f" /Type /XObject /Subtype /Image /Width {image.width}"
            f" /Height {image.height} /ColorSpace /DeviceRGB /BitsPerComponent 8"
        )

    def _font(self, index: int, stream: object) -> str:
        if index == 0:
            widths = " ".join(str(_width_of_code(c)) for c in range(32, 256))
            return (
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
                f" /Encoding /WinAnsiEncoding /FirstChar 32 /LastChar 255"
                f" /Widths [{widths}] >>"
            )
        table = self._fonts[index]
        lines = [
            f"<{code:02X}> <{''.join(f'{u:04X}' for u in _utf16(ch))}>"
            for ch, code in sorted(table.items(), key=lambda kv: kv[1])
        ]
        cmap = (
            "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            "/CMapName /LNTest def /CMapType 2 def\n"
            "1 begincodespacerange <00> <FF> endcodespacerange\n"
            f"{len(lines)} beginbfchar\n" + "\n".join(lines) + "\nendbfchar\n"
            "endcmap CMapName currentdict /CMap defineresource pop end end\n"
        )
        ref = stream("", cmap.encode(), False)  # type: ignore[operator]
        widths = " ".join(str(_DEFAULT_WIDTH) for _ in range(32, 256))
        return (
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
            f" /FirstChar 32 /LastChar 255 /Widths [{widths}]"
            f" /ToUnicode {ref} 0 R >>"
        )


def _utf16(ch: str) -> list[int]:
    raw = ch.encode("utf-16-be")
    return [int.from_bytes(raw[i : i + 2], "big") for i in range(0, len(raw), 2)]


# --- standard security handler, revision 3 (RC4, 128-bit) -----------------------------

_PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


def _rc4(key: bytes, data: bytes) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) % 256
        s[i], s[j] = s[j], s[i]
    out = bytearray()
    i = j = 0
    for byte in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(byte ^ s[(s[i] + s[j]) % 256])
    return bytes(out)


def _md5(data: bytes) -> bytes:
    return hashlib.md5(data, usedforsecurity=False).digest()


def _padded(password: str) -> bytes:
    return (password.encode("latin-1") + _PAD)[:32]


def _rc4_security(
    user: str, owner: str, file_id: bytes
) -> tuple[bytes, bytes, bytes, int]:
    p = -3904  # print/copy allowed bits; any value works for tests
    okey = _md5(_padded(owner))
    for _ in range(50):
        okey = _md5(okey)
    o = _rc4(okey, _padded(user))
    for i in range(1, 20):
        o = _rc4(bytes(b ^ i for b in okey), o)
    key = _md5(_padded(user) + o + struct.pack("<i", p) + file_id)
    for _ in range(50):
        key = _md5(key)
    u = _rc4(key, _md5(_PAD + file_id))
    for i in range(1, 20):
        u = _rc4(bytes(b ^ i for b in key), u)
    return o, u + b"\x00" * 16, key, p


def _object_key(key: bytes, number: int) -> bytes:
    return _md5(key + number.to_bytes(3, "little") + b"\x00\x00")[:16]


# --- conveniences ---------------------------------------------------------------------


def rasterise(path: Path, index: int = 0, scale: float = 2.0) -> tuple[bytes, int, int]:
    """RGB pixels of one page, for building scanned-style image pages."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    try:
        page = doc[index]
        try:
            bitmap = page.render(scale=scale, rev_byteorder=True)
            try:
                w, h, stride = bitmap.width, bitmap.height, bitmap.stride
                n = bitmap.n_channels
                buf = bytes(bitmap.buffer)
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        doc.close()
    rows = []
    for y in range(h):
        row = buf[y * stride : y * stride + w * n]
        if n == 4:
            row = b"".join(row[i : i + 3] for i in range(0, len(row), 4))
        rows.append(row)
    return b"".join(rows), w, h


def string(text: str) -> str:
    """A PDF literal string for raw dictionaries."""
    return _pdf_string(text)
