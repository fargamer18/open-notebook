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

### Recommended: the installer

1. Download into one folder: `install-litenotebook.sh`,
   `LiteNotebook-0.9.3-linux-x86_64.tar.gz` and, optionally, the semantic-search
   runtime `LiteNotebook-embedding-runtime-0.9.3-linux-x86_64.tar.gz`
   (needed to add new PDFs; ~1 GB installed).
2. Run it as your normal user (no `sudo`):
   ```bash
   bash install-litenotebook.sh
   ```
   It asks about the runtime, a menu entry and a desktop shortcut, checks the
   archives' SHA-256 (built into the installer), and installs to
   `~/.local/opt/litenotebook`. Start LiteNotebook from the application menu
   or with `litenotebook`.

Without questions: `bash install-litenotebook.sh --yes
[--with-embedding-runtime] [--prefix DIR] [--no-desktop-entry]
[--desktop-shortcut]`. Run a newer version's installer to upgrade; your data
stays. **Uninstall:** `litenotebook-uninstall` removes the program, launcher,
menu entry and icon, and keeps your notebooks. Details:
[docs/installers.md](docs/installers.md).

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

### Tarball (advanced)

```bash
tar -xzf LiteNotebook-0.9.3-linux-x86_64.tar.gz
cd LiteNotebook-0.9.3-linux-x86_64
./litenotebook                          # start
./litenotebook --install-desktop-entry  # optional: add to the app menu
```

## Windows 10 / 11 (x64) — prepared, not yet verified

Download `LiteNotebook-Setup-0.9.3-x64.exe` and run it. The setup wizard:

- installs for your user only, in `%LOCALAPPDATA%\Programs\LiteNotebook`,
  with no administrator rights (you can choose another folder);
- offers **[ ] Install local semantic-search runtime**: enables local
  embeddings and semantic search, needed to add new PDFs; approximately 1 GB
  installed;
- offers **[x] Start Menu shortcut** and **[ ] Desktop shortcut**;
- ends with **[x] Launch LiteNotebook**.

It downloads nothing and installs no AI model; LiteNotebook's own setup
wizard offers those later. Running a newer installer upgrades in place.
**Uninstalling removes the program and shortcuts only. Your notebooks in
`%LOCALAPPDATA%\LiteNotebook` are kept.** The installer is not code-signed
yet, so Windows may show a SmartScreen warning.

`LiteNotebook-0.9.3-windows-x64.zip` is a portable alternative.

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
| Embedding runtime (**packaged app only**) | The installers' optional component (Windows wizard; Linux `--with-embedding-runtime`); or Settings → Local AI → Semantic search: enter the path of the downloaded `LiteNotebook-embedding-runtime-0.9.3-…` archive → *Check and install*; or unpack it as an `embedding-runtime` folder next to the program | needed to index new PDFs in the packaged app (~330 MB download, ~1 GB unpacked) |
| Speech: sherpa-onnx + voices | Settings → Local AI → Speech | for reading audio overviews aloud |

Everything is checked against pinned SHA-256 checksums before use.

## Updating

- Install the new version over the old one: run the new installer (Linux
  `install-litenotebook.sh` or the Windows wizard), or replace the AppImage or
  tarball.
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
| Program | `~/.local/opt/litenotebook` (installer), or where you put the AppImage / tarball | `%LOCALAPPDATA%\Programs\LiteNotebook` |

To remove everything, delete the program, then the data folder (after making
a backup if you want one).
