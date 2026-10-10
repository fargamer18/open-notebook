# PDF engine licensing decision (release blocker)

> **Not legal advice.** This document collects license facts from upstream
> sources and technical measurements so that the project owner (with legal
> counsel where needed) can make a decision. It does not say any option is
> legally sufficient.

Status: **open — a decision is required before any public release.**
Nothing has been changed yet; LiteNotebook 0.9.x still uses PyMuPDF.

## 1. Current dependency

| | |
|---|---|
| Package | PyMuPDF 1.28.2 (MuPDF 1.28.2), imported as `pymupdf` |
| Where | core app (bundled in every desktop package); not in the embedding runtime |
| Size in the package | ≈ 53 MB of the 245 MB core app |

How LiteNotebook uses it today (all in three modules, ≈ 120 lines that touch
the PyMuPDF API):

| Use | Module | PyMuPDF API |
|---|---|---|
| Validate an upload (real PDF, not encrypted, has pages) | `ingest/pdf.py` | `open`, `is_pdf`, `needs_pass`, `page_count` |
| Page text for chunking/citations | `ingest/pdf.py` | `page.get_text("text")` |
| OCR input image for scanned pages (Tesseract) | `ingest/pdf.py` | `page.get_pixmap`, `Pixmap.save` |
| Reader page images (PNG, capped size) | `reader/render.py` | `get_pixmap(matrix)`, `tobytes("png")` |
| Word boxes for highlights, citation regions, annotation anchoring, evidence validation | `reader/geometry.py` | `get_text("words", sort=True)`, `rotation_matrix`, `Rect` |
| Page count fallback | `reader/render.py` | `page_count` |
| Build only: render the SVG icon to PNG/ICO | `tools/build.py` | `open(svg)`, `get_pixmap` |
| Tests: create synthetic PDFs | `tests/` (≈ 20 places) | `new_page`, `insert_text`, `save` |

LiteNotebook does **not** use PDF annotations, forms, JavaScript, editing or
saving of user PDFs. Highlights and annotations are LiteNotebook's own data
(SQLite), drawn over the rendered page image.

## 2. Current license facts (from upstream, checked 2026-10-10)

- PyPI metadata for PyMuPDF 1.28.2: *"Dual Licensed - GNU AFFERO GPL 3.0 or
  Artifex Commercial License"*; *"GNU AGPL v3. Free for open-source
  projects"*; *"separate commercial licences available from Artifex for
  proprietary applications."* (<https://pypi.org/project/PyMuPDF/>)
- PyMuPDF documentation, *About*: *"PyMuPDF and MuPDF are now available
  under both, open-source AGPL and commercial license agreements."* Users
  who cannot meet the AGPL requirements are told to contact Artifex for a
  commercial license; *"Artifex is the exclusive commercial licensing agent
  for MuPDF."* (<https://pymupdf.readthedocs.io/en/latest/about.html>)
- Artifex licensing page: *"If you can't meet the requirements of the GNU
  AGPLv3 above, a commercial license is required."* It also summarises AGPL
  obligations, including source disclosure to users who interact with the
  software over a network. (<https://artifex.com/licensing>)
- LiteNotebook's repository is MIT-licensed (inherited from Open Notebook).
  **The MIT license of LiteNotebook does not change the license of PyMuPDF.**
  Each bundled component keeps its own terms, and the desktop package
  bundles PyMuPDF.

## 3. Distribution implications to review

These are questions for review, not conclusions:

1. The desktop packages (AppImage, tarball, Windows installer/zip) **contain
   PyMuPDF/MuPDF binaries**. Distributing them means distributing AGPL-licensed
   code (unless a commercial license applies).
2. Under the AGPL route, the distributor would need to meet the AGPL's
   conditions for the distributed work, which commonly means offering the
   **complete corresponding source** of the application under AGPL-compatible
   terms, plus the license notices. Whether and how that applies to the whole
   LiteNotebook package is the core question to review.
3. MIT-licensed code can be combined into an AGPL-licensed whole, but the
   combined distributed work is then subject to the AGPL's conditions. MIT
   does not exempt the package from them.
4. LiteNotebook is a local app that serves its UI on 127.0.0.1 to the same
   user. How the AGPL's network clause (section 13) applies to that model
   should be part of the review; do not assume it does not.
5. Running from source on your own machine without distributing anything is
   a different situation from publishing packages.

## 4. Options

### A. Release LiteNotebook under AGPL-compatible terms

- Keep PyMuPDF and distribute the packages together with the complete
  corresponding source (the repository already contains it) under terms
  compatible with the AGPL. In practice this usually means licensing the
  distributed LiteNotebook application under the AGPL-3.0, or keeping MIT for
  the project's own code while acknowledging the combined package is AGPL.
- Cost: no engineering work. Requires a deliberate licensing decision by the
  copyright holders, updated LICENSE/NOTICE files and source offers next to
  each release.
- Effect: anyone redistributing or modifying the packages must also follow
  the AGPL; closed-source forks of the package are not possible.

### B. Obtain a commercial PyMuPDF/Artifex license

- Keep PyMuPDF, keep MIT for the project, and license MuPDF commercially from
  Artifex for distribution.
- Cost: commercial terms (price and scope are negotiated with Artifex; not
  known here). The license must cover the intended distribution (free public
  downloads, platforms, number of users, forks).
- No engineering work.

### C. Replace PyMuPDF with a permissively licensed stack

Technical feasibility only (no code was changed; the measurements below come
from a separate scratch environment).

#### Candidates

| | pypdfium2 + PDFium | pypdf (+ a renderer) | others |
|---|---|---|---|
| License | pypdfium2 Apache-2.0 / BSD-3-Clause; PDFium BSD-style (Google). Some PDFium dependencies carry other permissive licenses whose notices must ship with binaries | pypdf BSD-3-Clause | pdfminer.six (MIT): text + coordinates, no rendering; poppler (GPL), Ghostscript (AGPL), MuPDF (AGPL): not permissive |
| Version checked | pypdfium2 5.14.0 (PDFium 156.0.8076.0), released 2026-10-04 | pypdf 6.20.0 | |
| Wheels | Windows x64 (3.9 MB), Linux manylinux_2_17 x86_64 (3.8 MB), plus arm64, macOS, musl | pure Python | |
| Text extraction | yes (`PdfTextPage.get_text_bounded/range`) | yes (lower quality on complex layouts) | |
| Character / word boxes | **character** boxes (`get_charbox`); words must be built by LiteNotebook | via visitor callbacks, imprecise | pdfminer: good |
| Rendering | yes, fast (Chromium's engine) | **no** — needs a renderer, and the permissive choices are PDFium itself | |
| Encrypted PDFs | detects (error "Incorrect password") | detects | |
| Malformed PDFs | stricter: refused the truncated test PDF that MuPDF repaired | refused | |
| JavaScript / actions | pypdfium2 wheels are built without V8/XFA; actions are never run | not executed | |
| Thread safety | not thread-safe (same as PyMuPDF): needs the same process-wide lock | pure Python | |

**pypdfium2 is the only realistic drop-in.** pypdf cannot render pages, and
pdfminer.six cannot render either, so both would still need PDFium for the
Reader.

#### Measurements (this machine, same PDFs, both engines)

| Test | PyMuPDF 1.28.2 | pypdfium2 5.14.0 |
|---|---|---|
| Text per page (synthetic text PDFs) | 0.3–0.7 ms | 0.07–0.7 ms |
| Normalised text identical to PyMuPDF | — | yes on every synthetic PDF tested |
| Render 1.5× + PNG encode, A4 text page | ≈ 15 ms | ≈ 19–21 ms (PNG via Pillow) |
| Render 1.5× + PNG, scanned Maths pages (copies) | 54–61 ms | 57–60 ms |
| Word boxes matching PyMuPDF's (same text, IoU > 0.5), naive whitespace grouping | — | 60–67 %: PDFium boxes are tight glyph boxes, MuPDF uses line height; punctuation splits differ |
| Rotated page | correct via `rotation_matrix` | needs LiteNotebook's own rotation transform |
| Password-protected | detected | detected |
| `%PDF-` header but garbage | rejected | rejected |
| Truncated PDF (half the file) | **repaired, 3 pages read** | rejected |
| 14 400 × 14 400 pt page at 1.5× | refused by MuPDF's limit (`FzErrorLimit`) | rendered a 21 600² bitmap in 8.8 s — LiteNotebook's own pixel cap (`MAX_PIXELS_SIDE`) is essential |
| 2 000-page PDF | fine | fine |

Rendering also needs an image encoder for PNG output: Pillow (HPND license,
≈ 10 MB with its libraries), or LiteNotebook's own minimal PNG writer
(zlib, ~40 lines). Net package effect: about −53 MB (PyMuPDF) + 8 MB
(PDFium) [+ Pillow] ≈ **−35 to −45 MB**.

#### Estimated migration surface

| Area | Work |
|---|---|
| `ingest/pdf.py` | open/validate (error mapping), text per page, OCR render |
| `reader/geometry.py` | build words from character boxes (whitespace and line breaks), line/block ids, rotation transform; re-validate highlight matching against chunks extracted by MuPDF in existing notebooks (stored chunk text stays as is) |
| `reader/render.py` | render + PNG encoding; keep the pixel cap |
| Evidence validation (Study, Audio), annotations | rely on geometry; regression-test only |
| `tools/build.py` icons | commit pre-rendered PNG/ICO or render the SVG another way |
| Tests | ≈ 20 places create PDFs with PyMuPDF; switch to a small hand-written PDF generator or keep PyMuPDF as a **development-only** test dependency (never bundled) |
| Packaging | PyInstaller hook for pypdfium2 (exists in hooks-contrib), license notices for PDFium dependencies |

Rough size: 1–2 focused phases, most of it in geometry and highlight
regression testing. Risks: worse tolerance of damaged PDFs, different text
order on complex multi-column pages (affects new chunks only), highlight
quality on existing notebooks.

## 5. Recommendation / options summary

| | Engineering | Project license | Ongoing |
|---|---|---|---|
| A. AGPL-compatible release | none | changes (decision for the copyright holders) | ship source with every release |
| B. Artifex commercial license | none | stays MIT | commercial agreement |
| C. Replace with pypdfium2 | 1–2 phases | stays MIT | PDFium notices; lose MuPDF's PDF repair |

Technical recommendation, **if the goal is to keep MIT and avoid
commercial terms**: option C with pypdfium2 is feasible. It gives equal text
and similar rendering speed and a smaller package. The cost is the
word-geometry rewrite, highlight regression work and weaker handling of damaged
PDFs. If keeping MuPDF's robustness matters more than the license model,
choose A or B. **The choice is the project owner's; this phase does not
change the PDF engine.**

Not legal advice.
