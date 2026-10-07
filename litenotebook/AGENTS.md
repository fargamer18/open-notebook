# LiteNotebook — Agent Rules

LiteNotebook is a lightweight, local-first notebook RAG app for one laptop
(Ryzen 7 4800H, 16 GB RAM, GTX 1650 Ti 4 GB, data on an HDD). It is built on
the `litenotebook` branch next to the legacy Open Notebook code, which will be
retired only after LiteNotebook passes its own end-to-end tests.

**The repository-root `AGENTS.md` describes Open Notebook (SurrealDB, LangGraph,
Next.js). Its rules do not apply inside `litenotebook/`.** These do.

## Architecture (fixed — change only with a recorded decision)

```
Browser (Jinja2 + HTMX, vendored, no CDN)
  → FastAPI, single process, 127.0.0.1:5056
      ingest:    PyMuPDF text layer per page → Tesseract fallback for empty pages
      chunk:     custom page-aware chunker (MiniLM tokenizer counts)
      embed:     sentence-transformers/all-MiniLM-L6-v2, CPU, normalized, 384-d
      index:     FAISS IndexIDMap2(IndexFlatIP(384)), one per notebook
      retrieve:  FTS5 BM25 + FAISS → RRF (k=60) → near-duplicate filter → top k
      generate:  Qwen3-4B via OpenAI-compatible llama.cpp server, ONE call per question
      cite:      [S#] → source / page / chunk, persisted per message
  → SQLite (WAL, FTS5) = source of truth   data/notebooks/{id}/index.faiss = derived cache
```

## Hard rules

- **Forbidden in the runtime:** Docker, SurrealDB, PostgreSQL, Redis, Celery,
  Elasticsearch, LangChain, LangGraph, LlamaIndex, React, Next.js. A test
  (`tests/test_phase0_environment.py`) fails if any are installed.
- **Qwen is for generation only.** Embeddings always come from MiniLM in-process.
- **torch stays CPU-only** (pinned to the PyTorch CPU index in `pyproject.toml`).
  The GPU belongs to llama.cpp. No `nvidia-*` / `triton` packages.
- **Chunks never exceed 256 MiniLM tokens** (the model silently truncates beyond
  that). PDF chunks never cross a page boundary. Every chunk keeps source id,
  page, order within the source, character offsets within the page text,
  and Markdown heading context where applicable. Chunks overlap.
- **SQLite is authoritative.** Embeddings are stored as float32 BLOBs on the
  chunk row. FAISS uses `chunk.id` as its id, is written atomically (temp file
  + `os.replace`), is checked against SQLite (count, model, dimension) on load,
  and can always be rebuilt from SQLite without re-embedding.
- **One LLM call per normal question.** No multi-step agent chains.
- **Bind to loopback only.** No auth in the MVP; mutating requests must pass a
  Host/Origin check. No telemetry, no CDN assets.
- **Never import from `legacy/` or the Open Notebook packages** (`open_notebook`,
  `api`, `commands`). `legacy/localbook/` is reference material only.
- **Secrets:** never read, print, or commit the repository `.env` (it belongs
  to Open Notebook and its old values are considered compromised).
  LiteNotebook config comes from `LITENOTEBOOK_*` variables and the SQLite
  `settings` table.
- Concurrency: plain `def` routes / worker threads with stdlib `sqlite3` are
  fine; this project does not follow Open Notebook's async-first DB rule.

## Commands (run inside `litenotebook/`)

- Install: `uv sync`
- Tests: `uv run pytest` (opt-in: `-m model`, `-m llm`)
- Lint/types: `uv run ruff check .` · `uv run mypy src`

## Layout

- `src/litenotebook/` — the package (src layout, so `legacy/` and Open
  Notebook code can't be imported by accident)
- `tests/` — pytest suite
- `data/` — runtime data (gitignored): SQLite DB, per-notebook sources and FAISS index
