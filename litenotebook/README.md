# LiteNotebook

A lightweight, **local-first** research notebook. Add your PDFs to
notebooks, read and annotate them, search them, ask questions answered
with source / page citations, take notes, practise with quizzes and listen
to audio overviews. Everything runs on your own computer. No account, no
cloud service and no internet connection are needed for daily use.

> **Status: 0.9.2 pre-release — not a stable v1.** Open release blockers:
>
> 1. **Windows is Prepared only.** It is not yet Built, Automated-tested or
>    Manually verified on real Windows; see [docs/windows.md](docs/windows.md).
> 2. **Linux release builds** must come from the Ubuntu 22.04 (glibc 2.35)
>    build ([docs/linux.md](docs/linux.md)). Builds made on newer
>    distributions are development artifacts.
>
> PDFs are read with PDFium (pypdfium2); PyMuPDF was removed in 0.9.2, so the
> former PDF-licensing blocker is closed — see
> [docs/pdf-licensing-decision.md](docs/pdf-licensing-decision.md) and
> [docs/pdf-engine.md](docs/pdf-engine.md).

## Platforms

| | Linux x86_64 | Windows 11 / 10 x64 |
|---|---|---|
| Status | **Built and verified on Linux** (source mode and packaged app) | **Prepared**: code, installer script and CI job exist; nothing run on Windows yet |
| Package | AppImage, tar.gz (+ optional embedding runtime) | Inno Setup installer, zip (+ optional runtime component) |
| Minimum | glibc 2.35 (Ubuntu 22.04, Debian 12, Fedora 36 or newer) for the Ubuntu-built release | — |

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

See [INSTALL.md](INSTALL.md). In short:

- **Linux:** download the AppImage (or tarball), make it executable, run it.
- **Windows:** once built and verified, run the installer. It installs per
  user, needs no admin rights, and uninstalling keeps your notebooks.
- **From source:**
  - `uv sync`
  - `uv run litenotebook serve --open`

LiteNotebook opens in your default browser at `http://127.0.0.1:<port>`.
It listens on your own computer only.

## Your data

- Notebooks live in your user folder: `~/.local/share/LiteNotebook` on
  Linux, `%LOCALAPPDATA%\LiteNotebook` on Windows.
- Logs are kept separately. They contain no document text, notes, prompts or
  keys.
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
```

Configuration is through environment variables (`LITENOTEBOOK_DATA_DIR`,
`LITENOTEBOOK_PORT`, `LITENOTEBOOK_LLM_BASE_URL`, …); see
[docs/platform.md](docs/platform.md). Architecture and rules:
[AGENTS.md](AGENTS.md).

LiteNotebook's own code is MIT-licensed. Bundled third-party components keep
their own licenses ([docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md)).
