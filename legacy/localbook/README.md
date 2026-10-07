# LocalBook (legacy reference)

Verbatim copies of the command-line prototype at `/run/media/faraz/DATA/LocalBook/app/`,
preserved on 2026-10-07 before LiteNotebook reimplements it. The original
directory was not version-controlled.

**Reference material only.** Nothing under `litenotebook/` may import from,
read, or execute these files. They contain hardcoded absolute paths and are
not expected to run from this location.

Not copied: `.venv/` (5.9 GB, CUDA torch), `indexes/` (OCR SQLite DB, FAISS
index, metadata JSON), model files, caches, and the source PDFs.

| File | What it does | Carried into LiteNotebook as |
|---|---|---|
| `ocr_index.py` | PyMuPDF page render (1.5x) → `tesseract --psm 6` → SQLite `pages` + external-content FTS5 | OCR fallback for pages without a text layer |
| `build_embeddings.py` | all-MiniLM-L6-v2 on CPU, normalized → `faiss.IndexFlatIP` + positional JSON metadata | Embedder; replaced by per-chunk vectors in `IndexIDMap2` keyed by `chunk.id` |
| `search.py` | BM25 keyword search over FTS5 | Superseded by hybrid retrieval |
| `ask.py` | BM25 top 30 + FAISS top 30 → RRF (k=60) → Jaccard near-dup filter (0.8) → top 8 `[S#]` → Qwen3-4B via llama-server → citation resolution | Retrieval + RAG core |
| `inspect_pdfs.py` | Text-layer coverage report (GOOD / PARTIAL / OCR NEEDED) | Per-page "needs OCR" detection |

Known limitations fixed by LiteNotebook (see the Phase 0 audit):

- Whole pages are embedded, but MiniLM truncates input at 256 tokens, so most of each page never reaches the vector index.
- FAISS row position ↔ JSON list index mapping has no delete/update path.
- Every page is OCR'd even when a text layer exists.
- Page hash is `sha256(path:page)`, so edited files are never re-indexed.

## Checksums (identical to the originals at copy time)

```
44166b7dc00853db051daf280231b3ebcbc7d74f4db742fb9792aaec8cacdeed  ask.py
44f17b6f82770fbb31d38e992733693c6858baa9af945d63b53f4cc932033e80  build_embeddings.py
18f8fe259dfd1dee66a151bd4c2a2ac7769a83aec5fc69f8588f1a3836791afc  inspect_pdfs.py
ebceab9f075021f9335deb557911399f36dbb32557279cb9d0395b6af1d3ad51  ocr_index.py
b5e562c61a13f3bb75e67f8e5eff06e4cf409d8800a02e53ada519abcd401dcf  search.py
```
