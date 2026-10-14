"""Minimal PNG writer for rendered pages (8-bit RGB or gray, no filter).

PDFium returns raw pixels; the Reader and OCR need PNG files. A dozen lines
of zlib avoids an image library dependency.
"""

from __future__ import annotations

import struct
import zlib

PNG_LEVEL = 6  # zlib level: smaller cache files for a few ms more per page


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def encode_png(
    pixels: bytes | memoryview, width: int, height: int, stride: int, channels: int
) -> bytes:
    if channels not in (1, 3):
        raise ValueError("only gray or RGB pixels are supported")
    row = width * channels
    view = memoryview(pixels)
    raw = b"".join(
        b"\x00" + view[y * stride : y * stride + row].tobytes() for y in range(height)
    )
    color_type = 0 if channels == 1 else 2
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, PNG_LEVEL))
        + _chunk(b"IEND", b"")
    )
