# PDF engine (PDFium via pypdfium2)

Since 0.9.2 (Phase 17) LiteNotebook reads PDFs with **PDFium** through
**pypdfium2 5.14** (PDFium 156.0.8076.0). PyMuPDF was removed; the reasons
are in [pdf-licensing-decision.md](pdf-licensing-decision.md).

## The boundary

Everything that reads a PDF goes through `litenotebook.pdf`. No other module
imports a PDF library.

| Module | Role |
|---|---|
| `pdf/engine.py` | `open_pdf(path, name)`, `PDF_LOCK`, `engine_versions()` |
| `pdf/pdfium_engine.py` | `PdfiumDocument`: `page_count`, `page_size(i)`, `page_text(i)`, `page_words(i)`, `render(i, scale)`, `close()`; word reconstruction |
| `pdf/models.py` | `PageSize`, `WordBox`, `RenderedPage`, `PdfError` and its subclasses `PdfDamagedError`, `PdfPasswordError`, `PdfNoPagesError` |
| `pdf/png.py` | a small PNG writer (zlib), so no image library is needed |

Callers: upload validation and ingestion (`ingest/pdf.py`), OCR page images
(`ingest/pdf.py`), Reader page images and the page-count fallback
(`reader/render.py`), and word geometry for highlights, citations, annotations
and evidence checks (`reader/geometry.py`). The Diagnostics page reports
`engine_versions()`.

## Coordinate convention

All consumers use one convention, which is the same one the stored data has
always used (no stored row was rewritten):

* A page is what the Reader shows: the **crop box**, with the page's
  **`/Rotate` applied**.
* Word boxes, highlight rectangles and annotation rectangles are
  **normalised** to that displayed page: `0..1` on both axes, **origin
  top-left**, x to the right, y down, clamped to `0..1`.
* `page_size` is in PDF points (1/72 inch) of the displayed page, so a
  portrait page with `/Rotate 90` reports a landscape size.

Word boxes are mapped from PDF user space with the transform PDFium itself
uses for rendering (`FPDF_PageToDevice`), so the text layer and the page
image always agree. Tests check this at 0°, 90°, 180° and 270°, with offset
crop boxes, by comparing word boxes with the rendered ink.

Under PyMuPDF the same rectangles came from `get_text("words")` mapped through
`rotation_matrix` and divided by `page.rect`. Measured on generated pages at
all four rotations and with crop boxes, word centres agree with the old
values to within 0.0005 of the page, so stored annotation rectangles stay
where they were.

## Text and words

PDFium exposes characters, not words, and its own text order (built for
display) mixes lines on some rotated pages. LiteNotebook therefore takes from
PDFium only each character's code, box, origin, angle, font size and text
object, and rebuilds lines and words itself, deterministically:

1. **Order**: characters are ordered by the drawing position of their text
   object (Form XObjects expanded in place), then by PDFium's index. This is
   the order the document draws its text, which is what PyMuPDF used, so
   chunks extracted earlier still line up with the page.
2. **Lines**: measured in each character's own text direction. A new line
   starts when the baseline moves by more than half a line height, the text
   runs backwards, the direction changes, or the next text object starts more
   than 0.8 em after the previous glyph (table cells). Gaps inside one text
   object (justified `TJ` spacing) never split a line.
3. **Words** end at a space or tab, a new line, a gap of more than 0.15 em
   from the previous glyph to the next glyph's origin (MuPDF's word-space
   threshold), where PDFium generated a space between two neighbouring glyphs
   (it knows the font's space width), or where a text object that yields no
   characters is drawn in between (producers that draw each space as its own
   object). Text at angles that are not a multiple of 90° only splits at real
   spaces.
4. **Hyphens**: PDFium's end-of-line hyphen marker becomes `-` again, and
   page text keeps `-\n`, so the ingestion normaliser re-joins hyphenated
   words exactly as before.

`page_text` is the lines joined with `\n` and the words with one space; the
normaliser in `ingest/pdf.py` is unchanged.

### Compatibility with PyMuPDF (measured before PyMuPDF was removed)

Normalised page text, PyMuPDF vs PDFium:

| Set | Pages | Byte-identical | Token similarity |
|---|---|---|---|
| Generated fixtures (0/90/180/270°, crop boxes, Unicode, hyphenation, repeated phrases, mixed sizes, rotated text, annotations) | 17 | 17 | 1.000 |
| Seven real PDFs (papers, a long guide, reports, a poster) | 58 | 41 | min 0.977, mean ≥ 0.996 per file |
| Scanned Maths PDFs (copies) | 125 | 125 (no text layer) | — |

Every remaining difference is whitespace or line structure; the word
sequence is the same except for spacing around math operators. Extracted
characters per document agree to within 0.05 %.

Highlights (chunk → page → rectangles):

| Chunks highlighted on the 7 real PDFs | PyMuPDF Reader (0.9.1) | PDFium (0.9.2) |
|---|---|---|
| Chunks extracted by PyMuPDF (existing notebooks) | 50 / 277 | **277 / 277** |
| Chunks extracted by PDFium (new ingestion) | — | **271 / 271** |

The old Reader sorted words by position while chunks were in drawing order,
so multi-column pages often fell back to "highlight unavailable". Both now
use drawing order. Text that is not on the page is still never highlighted.

## Intentional and known differences

* **Damaged PDFs are rejected, not repaired.** MuPDF repairs some truncated
  or broken files; PDFium refuses more of them. Such an upload is rejected
  with *"‹name› appears damaged or unsupported. Re-save or export it as a PDF
  again (for example with Print to PDF) and upload the new file."* No repair
  is attempted. Sources ingested earlier from such a file keep their stored
  text and chunks; only the Reader may not be able to show their pages.
* **Encrypted PDFs** (user password) are rejected as before ("is
  password-protected"); there is no password entry. An unsupported security
  handler gets its own message.
* **Text differences** (whitespace only): U+00A0 (no-break space) is read as a
  normal space; U+200B (zero-width space) is dropped; sub/superscripts in
  equations stay on one line; a few table cells are separated by a space
  instead of a line break; a space can appear after large math operators.
* **Rendering** is pixel-for-pixel different (anti-aliasing) but not
  different in content: on generated pages the mean pixel difference is under
  1 %, with the same page size, orientation and crop.
* **Rotated text** (vertical labels, tilted words) is kept in drawing order
  and grouped into words; PyMuPDF's word order there was by position.
* Right-to-left scripts follow drawing order, as with PyMuPDF.

## Rendering

`render(i, scale)` draws the displayed page (crop box, rotation) on white, with
annotation appearances and without form fields, and returns a PNG. Callers cap
the long side (Reader: 4000 px, OCR: 4000 px at up to 2×); the engine refuses
anything over 40 megapixels as a second line of defence.

## Safety

* Only parsing, text extraction and rasterisation are used. Forms are never
  initialised and the pypdfium2 wheels contain no JavaScript (V8) or XFA
  engine, so document JavaScript, actions, launch actions, links and embedded
  files are never executed, followed or extracted. The hostile-PDF tests
  check that no process starts and no network connection is attempted.
* PDFium's "last error" is process-wide and not set by every failure; the
  engine resets it before each load, so a damaged file is never reported as
  password-protected because of an earlier file.
* Error messages name the file, never its path.

## Threads and resources

PDFium is not thread-safe, not even across different documents
(pypdfium2 documentation), so the lock was kept: every PDFium call, including
closing objects, holds the process-wide `PDF_LOCK`. Documents take it per
operation, so a long ingestion never blocks the Reader for more than one
page. A test wraps PDFium's native entry points and checks that no two calls
ever overlap while ingestion, two geometry readers and a renderer run
concurrently. Removing the lock would need per-process isolation, which this
phase does not add.

Documents, pages, text pages and bitmaps are closed explicitly (context
managers and `finally`); a test counts PDFium's native open/close calls
through ingestion, rendering, geometry and highlight lookup and requires them
to balance, then deletes the source file (which fails on Windows if anything
still holds it).

## Performance (this machine; PyMuPDF measured before removal)

| | PyMuPDF 1.28.2 | PDFium (pypdfium2 5.14) |
|---|---|---|
| Open document | 0.3–335 ms | 0.3–48 ms |
| Text, dense two-column page | ≈ 5 ms | ≈ 41 ms |
| Text, ordinary page | 1–3 ms | 11–21 ms |
| Reader words, dense page | ≈ 11 ms | ≈ 46 ms |
| Render 1.5× + PNG, scanned Maths pages | 3.8 s / 59 pages | 2.8 s / 59 pages |
| Extract + words + render, 6-page paper | 0.92 s | 1.24 s |
| Peak memory, same workload, scanned 59-page PDF | 334 MB | 68 MB |
| Peak memory, 6-page paper | 74 MB | 52 MB |
| `import` cost (RSS) | 50 MB | 21 MB |

Text extraction is slower because word reconstruction runs in Python
(character data comes through ctypes). It is a real regression for text-heavy
ingestion (≈ 40 s more per 1 000 dense pages) and for the first Reader text
layer of a page (tens of milliseconds); it is small next to embedding time and
within the Reader's budget. Rendering is faster and memory use is much lower.
