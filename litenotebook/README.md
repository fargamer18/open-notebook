# LiteNotebook

A lightweight, local-first alternative to Open Notebook: upload PDFs and text
into notebooks, search them, and ask questions answered by a local Qwen model
with source / page / chunk citations.

**Status:** Phase 0 (scaffold and environment only). No ingestion, retrieval,
or web UI yet.

## Requirements

- Python 3.12 (managed by `uv`)
- `tesseract` (OCR fallback for scanned pages)
- A running OpenAI-compatible `llama-server` with Qwen3-4B (generation only)
- `sentence-transformers/all-MiniLM-L6-v2` in the Hugging Face cache (downloaded on first use)

## Setup

```bash
cd litenotebook
uv sync          # CPU-only torch; no CUDA packages
uv run pytest
```

Optional: the repository lives on an HDD, so a cold `import torch` from a venv
there is slow. To keep the venv on the NVMe drive instead:

```bash
export UV_PROJECT_ENVIRONMENT=~/.local/share/litenotebook/venv
uv sync
```

## Configuration

Environment variables (all optional):

| Variable | Default |
|---|---|
| `LITENOTEBOOK_DATA_DIR` | `litenotebook/data` |
| `LITENOTEBOOK_HOST` / `LITENOTEBOOK_PORT` | `127.0.0.1` / `5056` (loopback only) |
| `LITENOTEBOOK_LLM_BASE_URL` | `http://127.0.0.1:8080/v1` |
| `LITENOTEBOOK_LLM_MODEL` | `Qwen/Qwen3-4B-GGUF:Q4_K_M` |
| `LITENOTEBOOK_CHUNK_TOKENS` / `LITENOTEBOOK_CHUNK_OVERLAP_TOKENS` | `200` / `40` (max 256) |
| `LITENOTEBOOK_TOP_K` | `8` |

Architecture and rules: [AGENTS.md](AGENTS.md).
