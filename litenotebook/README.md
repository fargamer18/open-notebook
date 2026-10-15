# LiteNotebook

A lightweight, **local-first** research notebook. Add your PDFs to
notebooks, read and annotate them, search them, ask questions answered
with source / page citations, take notes, practise with quizzes and listen
to audio overviews. Everything runs on your own computer. No account, no
cloud service and no internet connection are needed for daily use.

> **Status: 0.9.3 pre-release — not a stable v1.** The open release blocker
> is **Windows: Prepared only.** It is not yet Built, Automated-tested or
> Manually verified on real Windows; see [docs/windows.md](docs/windows.md)
> and [docs/release-checklist.md](docs/release-checklist.md).
>
> Linux release builds come from Ubuntu 22.04 (glibc 2.35) and are verified
> ([docs/linux.md](docs/linux.md)). Builds made on newer distributions are
> development artifacts.
>
> PDFs are read with PDFium (pypdfium2); PyMuPDF was removed in 0.9.2, so the
> former PDF-licensing blocker is closed — see
> [docs/pdf-licensing-decision.md](docs/pdf-licensing-decision.md) and
> [docs/pdf-engine.md](docs/pdf-engine.md).

## Platforms

| | Linux x86_64 | Windows 11 / 10 x64 |
|---|---|---|
| Status | **Built and verified on Linux** (source mode and packaged app; tested on Ubuntu 22.04 and Arch) | **Prepared**: code, installer script and CI job exist; nothing run on Windows yet |
| Package | AppImage (needs FUSE 2, or `--appimage-extract-and-run`), tar.gz (+ optional embedding runtime) | Inno Setup installer (unsigned), zip (+ optional runtime component) |
| Minimum | glibc 2.35 for the Ubuntu-built release; other glibc 2.35+ distributions expected, not individually tested | — |

macOS is not a target.

## What it does

- **Notebooks and sources:**
  - PDFs (OCR for scanned pages with Tesseract, if installed);
  - a built-in Reader with page images, highlights, annotations and a
    resume position.
- **Search:** fast keyword search everywhere, plus semantic (hybrid) search
  when the embedding component is installed.
- **Chat:**
  - answers from your sources, with citations that open the Reader at the
    passage;
  - a memory of earlier answers;
  - one answer at a time, with stop and retry.
- **Notes** linked to passages, answers and annotations.
- **Knowledge graph:** a Global and a Local view of concepts, notes,
  sources, conversations and corrections, with an inspector that opens the
  real note, chat or PDF passage. No AI needed ([docs/graph.md](docs/graph.md)).
- **Study:**
  - quizzes generated from the selected material and validated against it,
    with attempts, hints, review and concept progress;
  - **Audio Overviews**, two-host or narrated scripts grounded in your
    sources, read aloud by an optional local speech engine.
- **Backup:** notebook export/import, full backup and verified restore.
  Archives are portable between Linux and Windows.

## Optional local AI components

The app itself has no AI models. A setup wizard (and Settings → Local AI)
offers three optional components, downloaded only when you choose them:

| Component | What for | Size |
|---|---|---|
| Generation: llama.cpp + a Qwen3 GGUF model (or your own server / file) | Chat, quizzes, audio scripts | 1–5 GB |
| Embeddings: MiniLM (+ the embedding runtime in the packaged app) | Indexing new PDFs, hybrid search | ~90 MB (+ ~330 MB runtime download) |
| Speech: sherpa-onnx + Piper voices | Reading audio overviews aloud | ~60 MB |

Without them, LiteNotebook still opens and works for reading, notes,
keyword search, existing quizzes and audio, and backups. Indexing *new* PDFs
needs the embeddings component.

## Install

**Windows 10 / 11 (x64)** *(prepared; not yet verified on Windows)*

1. Download `LiteNotebook-Setup-0.9.3-x64.exe`.
2. Run the setup wizard. It installs for your user only (no administrator
   rights) and ends with *Launch LiteNotebook*.

**Linux (x86_64, glibc 2.35+)**

- **Recommended:** download `install-litenotebook.sh` and
  `LiteNotebook-0.9.3-linux-x86_64.tar.gz` into one folder, then run
  `bash install-litenotebook.sh` (no `sudo`). Start LiteNotebook from the
  application menu.
- **Alternative:** the single-file `LiteNotebook-0.9.3-x86_64.AppImage`
  (needs FUSE 2, or run it with `--appimage-extract-and-run`).
- **Advanced:** the plain `LiteNotebook-0.9.3-linux-x86_64.tar.gz`.

**Optional semantic-search runtime.** To add new PDFs, LiteNotebook needs its
local semantic-search runtime (approximately 1 GB installed). Tick it in the
Windows wizard, or put `LiteNotebook-embedding-runtime-0.9.3-linux-x86_64.tar.gz`
next to the Linux installer and say yes (or install it later in Settings →
Local AI). AI models and voices are never part of an installer: on first
start, LiteNotebook's setup wizard checks your hardware and offers them,
downloading only what you approve.

Uninstalling keeps your notebooks. LiteNotebook opens in your browser at
`http://127.0.0.1:<port>` and listens on your own computer only. Details:
[INSTALL.md](INSTALL.md) and [docs/installers.md](docs/installers.md).

## Your data and privacy

- **Local-first:** notebooks live in your user folder
  (`~/.local/share/LiteNotebook` on Linux, `%LOCALAPPDATA%\LiteNotebook` on
  Windows). The app listens on `127.0.0.1` only and makes no network request
  in daily use: no account, no telemetry, no update check.
- **Downloads only when you choose them:** optional components come from
  their publishers over HTTPS, pinned by version and SHA-256.
- **Logs** are kept separately. They contain no document text, notes,
  prompts, search text or keys, and your home folder is masked.
- **Deletion** removes the files, text, vectors and caches (SQLite
  `secure_delete` is on).
- **Backups:** notebook export/import, full backup and verified restore.
- See [docs/backup.md](docs/backup.md) and [docs/security.md](docs/security.md).

## Limitations (0.9.x)

- Indexing new PDFs needs the embeddings component. In the packaged app
  that means the separate embedding runtime download (PyTorch, ~1 GB
  unpacked).
- One generation (chat, quiz or audio) at a time. Answers are only as good as
  the local model, and Chat cannot prove a free-text answer ignored planted
  instructions in a document (Quiz and Audio validate theirs).
- No login on the local UI. On a shared computer, other local accounts could
  reach the loopback port while the app runs ([docs/security.md](docs/security.md)).
- No auto-update; install a new version over the old one.
- English-oriented speech voices only.

## Documentation

- **Installing and running:** [INSTALL.md](INSTALL.md), [docs/linux.md](docs/linux.md),
  [docs/windows.md](docs/windows.md), [docs/platform.md](docs/platform.md)
- **Features:** [docs/local-ai.md](docs/local-ai.md),
  [docs/study-quizzes.md](docs/study-quizzes.md),
  [docs/audio-overviews.md](docs/audio-overviews.md),
  [docs/backup.md](docs/backup.md)
- **Security and performance:** [docs/security.md](docs/security.md),
  [docs/performance.md](docs/performance.md)
- **Packaging and release:** [docs/packaging.md](docs/packaging.md),
  [docs/installers.md](docs/installers.md),
  [docs/pdf-licensing-decision.md](docs/pdf-licensing-decision.md),
  [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md),
  [docs/windows-test-plan.md](docs/windows-test-plan.md),
  [docs/release-checklist.md](docs/release-checklist.md)

## Development

```bash
cd litenotebook
uv sync                     # CPU-only torch; no CUDA packages
uv run pytest -q
uv run ruff check . && uv run mypy --strict src
uv run litenotebook serve --open
uv run python tools/browser_smoke.py   # release browser smoke (Chromium/Chrome/Edge)
```

Release tools: `tools/build.py` (packages), `tools/packaged_smoke.py`
(a built package), `tools/browser_smoke.py` (critical pages in a real
browser, source or package), `tools/release_stress.py` (stress, offline,
lifecycle); see [docs/packaging.md](docs/packaging.md).

Configuration is through environment variables (`LITENOTEBOOK_DATA_DIR`,
`LITENOTEBOOK_PORT`, `LITENOTEBOOK_LLM_BASE_URL`, …); see
[docs/platform.md](docs/platform.md). Architecture and rules:
[AGENTS.md](AGENTS.md).

LiteNotebook's own code is MIT-licensed. Bundled third-party components keep
their own licenses ([docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md)).
