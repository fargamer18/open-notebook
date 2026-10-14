"""Engine-independent values returned by the PDF boundary.

Coordinate convention (all engines, all consumers):

* A page is what the Reader displays: the **crop box** with the page's
  ``/Rotate`` applied.
* Word boxes and annotation rectangles are **normalised** to that displayed
  page: ``0..1`` on both axes, origin at the **top-left**, x to the right,
  y downwards. Values are clamped to ``0..1``.
* Page sizes are in PDF points (1/72 in) of the displayed page, so a
  portrait page with ``/Rotate 90`` reports a landscape size.

Stored annotation rectangles (schema 8+) already use this convention, so no
stored row is ever rewritten when the engine changes.
"""

from __future__ import annotations

from dataclasses import dataclass


class PdfError(ValueError):
    """A PDF could not be used. Messages name the file, never its path."""


class PdfDamagedError(PdfError):
    pass


class PdfPasswordError(PdfError):
    pass


class PdfNoPagesError(PdfError):
    pass


@dataclass(frozen=True)
class PageSize:
    width: float  # points, displayed (rotated, cropped) page
    height: float
    rotation: int  # the page's /Rotate (0, 90, 180, 270)


@dataclass(frozen=True)
class WordBox:
    """One word on the displayed page, normalised 0..1 (top-left origin)."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    block: int
    line: int


@dataclass(frozen=True)
class RenderedPage:
    png: bytes
    width: int  # pixels
    height: int
