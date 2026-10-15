# Installing LiteNotebook 0.9.3 (pre-release)

LiteNotebook is a desktop app that opens in your web browser. It needs no
Python, no command line and no internet connection after installation.
AI features are optional extras you install from inside the app.

Release status: see [README.md](README.md). Windows packages are **not yet
built or verified on Windows**. The Linux instructions below are verified on
Ubuntu 22.04 (clean container) and Arch Linux.

## Linux (x86_64)

Requirements: glibc 2.35 or newer (release builds are made on Ubuntu 22.04).
Tested on Ubuntu 22.04 and Arch Linux. Other distributions with glibc 2.35+
(for example Debian 12+ or Fedora 36+) are expected to work but are not
individually tested.

### AppImage (single file)

1. Download `LiteNotebook-0.9.3-x86_64.AppImage` and `SHA256SUMS`.
2. Optional, to check the download: `sha256sum -c SHA256SUMS --ignore-missing`
3. Make it executable and start it:
   ```bash
   chmod +x LiteNotebook-0.9.3-x86_64.AppImage
   ./LiteNotebook-0.9.3-x86_64.AppImage
   ```

Your browser opens LiteNotebook. The AppImage needs the FUSE 2 library
(`libfuse.so.2`). Ubuntu 22.04 and newer do not install it by default:
install `libfuse2` (Ubuntu 24.04: `libfuse2t64`), or run
`./LiteNotebook-0.9.3-x86_64.AppImage --appimage-extract-and-run`, or use the
tarball. Without FUSE 2 a direct start fails with "No suitable fusermount
binary found".

### Tarball

```bash
tar -xzf LiteNotebook-0.9.3-linux-x86_64.tar.gz
cd LiteNotebook-0.9.3-linux-x86_64
./litenotebook                          # start
./litenotebook --install-desktop-entry  # optional: add to the app menu
```

## Windows 10 / 11 (x64) — prepared, not yet verified

The installer `LiteNotebook-0.9.3-Setup.exe`:

- installs for your user only, in `%LOCALAPPDATA%\Programs\LiteNotebook`,
  with no administrator rights;
- adds a Start menu entry and, optionally, a desktop shortcut;
- has an optional "semantic search runtime" component.

The `.zip` is a portable alternative. **Uninstalling removes the program
only. Your notebooks in `%LOCALAPPDATA%\LiteNotebook` are kept.**

These packages come from the Windows CI job. Until that job has run and
someone has completed [docs/windows-test-plan.md](docs/windows-test-plan.md),
treat Windows as untested.

## First start

1. LiteNotebook opens `http://127.0.0.1:<port>` in your default browser. It
   only listens on your own computer.
2. The **setup wizard** checks your hardware and offers the optional
   components. Each one can be skipped and installed later from Settings →
   Local AI.
3. Create a notebook and add PDFs.

Starting LiteNotebook again while it runs just opens the same page.
**Quit:** Settings → Diagnostics → *Quit LiteNotebook*. Closing the browser
tab does not stop it.

## Optional components

| Component | Where | Notes |
|---|---|---|
| Generation (llama.cpp + Qwen3, or your own llama-server / GGUF file) | Setup wizard or Settings → Local AI | the download size is shown before you start |
| Embeddings: MiniLM | same | ~90 MB |
| Embedding runtime (**packaged app only**) | Settings → Local AI → Semantic search: enter the path of the downloaded `LiteNotebook-embedding-runtime-0.9.3-…` archive → *Check and install*; or unpack it as an `embedding-runtime` folder next to the program; or the Windows installer's optional component | needed to index new PDFs in the packaged app (~330 MB download, ~1 GB unpacked) |
| Speech: sherpa-onnx + voices | Settings → Local AI → Speech | for reading audio overviews aloud |

Everything is checked against pinned SHA-256 checksums before use.

## Updating

- Install the new version over the old one: replace the AppImage or
  tarball, or run the new installer.
- Your data is never touched by an update. If the database format changes,
  LiteNotebook first keeps a copy in `<data>/backups/`.
- There is no automatic update check.

## From source (developers)

```bash
git clone … && cd litenotebook
uv sync
uv run litenotebook serve --open
```

Running from a checkout keeps data in `litenotebook/data` (development
mode). See [docs/platform.md](docs/platform.md).

## Where things are

| | Linux | Windows |
|---|---|---|
| Notebooks and settings | `~/.local/share/LiteNotebook` | `%LOCALAPPDATA%\LiteNotebook` |
| Logs | `~/.local/state/LiteNotebook/logs` | `%LOCALAPPDATA%\LiteNotebook\Logs` |
| Program | where you put the AppImage / tarball | `%LOCALAPPDATA%\Programs\LiteNotebook` |

To remove everything, delete the program, then the data folder (after making
a backup if you want one).
