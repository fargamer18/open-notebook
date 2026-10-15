# Desktop packaging and distribution

LiteNotebook ships as a normal desktop application that opens in the
default browser. Users need neither Python nor the command line.

| Platform | Artifacts | Status |
|---|---|---|
| Linux x86_64 | `install-litenotebook.sh` (per-user installer, recommended) + `LiteNotebook-<v>-linux-x86_64.tar.gz`; `LiteNotebook-<v>-x86_64.AppImage` | **release build: Ubuntu 22.04 container (needs glibc ≥ 2.35)**, verified on Arch and in Ubuntu 22.04; development builds on Arch need glibc ≥ 2.38 |
| Windows x64 | `LiteNotebook-Setup-<v>-x64.exe` (Inno Setup wizard), `LiteNotebook-<v>-windows-x64.zip` (portable) | **Prepared** only: not Built, not tested, not verified ([windows.md](windows.md)) |
| Both | `LiteNotebook-embedding-runtime-<v>-linux-x86_64.tar.gz` / `-windows-x64.zip` (optional) | Linux built and verified; Windows prepared |

Every build also writes `SHA256SUMS`, `build-report.json` and
`release-manifest.json` (per artifact: file name, version, platform,
architecture, component type, size and SHA-256; no local paths), and
`tools/verify_release.py` checks the folder before upload. Installers:
[installers.md](installers.md).

Version: `litenotebook.__version__` (currently **0.9.3**, a pre-release).
`pyproject.toml` must match it; a test and the build both enforce this.

## Release blockers (0.9.x is not a public v1 release)

The packaging architecture is approved. These must be resolved before any
public v1 release:

1. **Windows is prepared/simulated, NOT runtime verified.** The Inno Setup
   script and the Windows CI job exist, but no Windows package has been built
   or run on a real Windows machine.
2. **Linux release baseline.** Production Linux artifacts must be built on
   Ubuntu 22.04 (glibc 2.35), for example with the CI workflow. The
   Arch-built artifact needs glibc ≥ 2.38 and is a development artifact only,
   not the broad-release Linux build.

## Build

```
uv sync --group packaging                 # pinned PyInstaller 6.22.3 (+ hooks 2026.7)
uv run --group packaging python tools/build.py [--skip-worker] [--skip-appimage]
uv run python tools/packaged_smoke.py dist/litenotebook [--runtime-archive FILE]
uv run python tools/browser_smoke.py --app PATH/TO/litenotebook[.exe]   # real browser

packaging/linux/build-in-ubuntu-2204.sh   # Linux RELEASE build (podman/docker, Ubuntu 22.04)
python tools/release_stress.py lifecycle dist/litenotebook --runtime-archive FILE
```

**Linux release builds** come from `packaging/linux/build-in-ubuntu-2204.sh`
or the CI job:
- the Ubuntu 22.04 image is pinned by digest, uv by version and SHA-256, and
  dependencies by `uv.lock` (`--frozen`);
- the source tree is copied into a work folder, so the repository is never
  written;
- measured on 0.9.x: the core app and the runtime need **glibc 2.35** and
  **GLIBCXX 3.4.30** (the newest symbol versions in the bundled
  `libgcc_s`/`libstdc++`).

**Every build writes `build-report.json`:**
- app, version, platform, architecture, git commit (`+dirty` if LiteNotebook
  files changed);
- Python version and implementation, PyInstaller version;
- the runner: OS, kernel, libc, distribution, CI runner name. No host names,
  user names or absolute paths.

**Every build also writes:**
- `glibc` (Linux): the minimum glibc / GLIBCXX and the files that need it;
- `artifacts`: each file with its size and SHA-256, also written to
  `SHA256SUMS`;
- `litenotebook-build.json` inside the app folder: version, platform,
  architecture, git commit, build time, and the expected runtime archive's
  name, size and SHA-256.

- Output goes to `dist/litenotebook/` and intermediate files to
  `build/package/`. Both are ignored by git; built binaries are never
  tracked.
- The AppImage step downloads appimagetool 1.9.1 from a pinned URL and
  verifies its SHA-256.
- On Windows the build runs Inno Setup (`ISCC`) when it is installed.
- `.github/workflows/litenotebook-build.yml` (repository root) builds both
  platforms on `ubuntu-22.04` and `windows-2022` when started by hand or by a
  `v*` tag: tests, package checks, packaged/browser/installer smoke tests,
  checksums and a combined release manifest, uploaded as workflow artifacts.
  It never publishes a GitHub Release.

## Why PyInstaller

| Option | Verdict |
|---|---|
| **PyInstaller** 6.22.3 (one-folder) | **chosen** |
| Nuitka | compiles everything to C: slow builds, more moving parts for the native wheels; no size win here (the size is in native libraries, not Python code) |
| Briefcase / cx_Freeze / PyOxidizer | smaller ecosystems for these wheels; PyOxidizer is unmaintained |
| Electron / Tauri | excluded: no Node, no second UI runtime |

**Why PyInstaller fits:**

- It is mature, with Windows and Linux support.
- It has maintained hooks for pypdfium2, numpy, torch and transformers.
- It handles the native extensions in use: SQLite/FTS5 (CPython),
  PDFium (pypdfium2), FAISS and tokenizers.
- Builds are fast and the folder layout is predictable.
- Its GPL license has an exception that allows distributing built
  applications.

**One-folder, not one-file:** the app starts in about 1 s instead of
unpacking a few hundred MB to a temp folder on every launch.

## Layout and components

```
LiteNotebook-0.9.2-linux-x86_64/          (Windows: installed to %LOCALAPPDATA%\Programs\LiteNotebook)
  litenotebook            (LiteNotebook.exe)   web app + launcher; NO torch
  _internal/              Python 3.12, libraries, templates, static files, migrations, catalogs
  litenotebook-build.json version, platform, build time, expected embedding runtime (name/size/SHA-256)
  embedding-runtime/      OPTIONAL, beside the app: litenotebook-embed-worker + its own _internal/
  litenotebook.png/.svg, litenotebook.desktop (Linux), README.txt, LICENSE.txt, THIRD_PARTY_LICENSES.md
```

**Component model (same as Phase 12/14)**

- **Core app:** always installed.
- **Embedding runtime:** optional download.
  - It is the MiniLM worker executable (PyTorch CPU, sentence-transformers,
    transformers, plus scipy/scikit-learn, which sentence-transformers
    imports). It speaks the same stdin/stdout protocol, is a separate
    process, shuts down when idle, and the parent never imports torch.
  - It is found, in order, at `LITENOTEBOOK_EMBED_WORKER`, then
    `embedding-runtime/` beside the app (the Windows installer component,
    or unpack it there), then `<data>/runtime/embedding/<version>/`.
  - That last location is filled by Settings → Local AI → "Install from
    file". The install checks the exact file name, size and SHA-256 against
    `litenotebook-build.json`, then extracts with the hostile-archive rules.
- **MiniLM model, llama.cpp + Qwen, speech + voices:** unchanged explicit
  downloads from the setup wizard or Settings → Local AI.

**Never bundled:** GGUF models, MiniLM files, llama.cpp, the speech engine,
voices, espeak-ng, notebook data and the render cache. The build manifest
records `"bundled_optional_components": []`.

**Size trade-off (measured, Linux x86_64)**

| | Unpacked | Download |
|---|---|---|
| Core app | 197 MB | 71 MB tar.gz / 67 MB AppImage |
| Embedding runtime | 992 MB | 330 MB tar.gz |
| Both in one package | ~1.2 GB | ~400 MB |

**Largest parts of the core app:**

- PDFium (pypdfium2): 8 MB (PyMuPDF, 55 MB, was removed in 0.9.2)
- FAISS + its libraries: 64 MB
- numpy + OpenBLAS: 41 MB
- libpython: 31 MB
- tokenizers and hf_xet: 11 MB each

**Largest parts of the runtime:**

- torch: 643 MB
- scipy: 76 MB
- transformers: 46 MB
- scikit-learn: 16 MB

Splitting keeps the base download small, and keeps the PyTorch download
for users who index PDFs on this machine.

**Limitation:** indexing new PDFs needs the embedding runtime **and** the
MiniLM model, because chunking uses MiniLM's tokenizer and every chunk is
embedded. Without them, a packaged app still opens, reads notebooks
imported from a backup (which carry their embeddings) and offers notes,
keyword search, quizzes and audio. A new PDF fails with "The semantic
search runtime is not installed…" and a pointer to Settings.

**Host libraries:** PyInstaller resolves shared libraries through the build
machine. On this Arch host it initially pulled the system's own
`/usr/lib/libtorch_cpu.so` (an MPI build), Intel oneAPI MKL from
`/opt/intel` and their MPI/UCX dependencies into the worker, which was
1.4 GB and mixed two torch builds. `packaging/pyinstaller/_binaries.py`
drops such host copies (108 libraries), so only the wheels' own libraries
ship.

The packaged worker produces **bit-identical embeddings** to source mode
(max |Δ| = 0.0 on a test pair).

**glibc:** most bundled libraries need glibc ≤ 2.27. The exceptions are
the `libstdc++`/`libgcc_s` copied from the build machine:

- Arch build: **glibc ≥ 2.38**;
- Ubuntu 22.04 build: **glibc ≥ 2.35** (measured by `tools/build.py`, which
  scans every bundled ELF file).

Release artifacts must come from the Ubuntu 22.04 build.

## Launch model (`litenotebook/desktop.py`)

1. A per-user log file starts (see Logs).
2. If the data folder's instance lock is held by a LiteNotebook server, the
   launcher reads its URL from the lock file, checks it at `/livez`, opens
   that page and exits. A second launch never starts a second server and
   never kills anything. If the lock is held but nothing answers (a CLI
   command, or a hung process), the user gets a clear error.
3. Port: the configured one (5056) if nothing listens there, otherwise a
   free loopback port chosen by the OS. The program already listening is
   never touched.
4. The app is served on **127.0.0.1 only**. Nothing listens on the network,
   so no firewall rule is needed or created.
5. When `/livez` answers, the default browser opens the URL (the
   `webbrowser` module, which honours `BROWSER`).
6. **Settings → Diagnostics → Quit LiteNotebook** (a POST, same-origin
   only) stops the server. Ctrl+C or logging out does the same. On exit the
   app stops only what it started: the embedding worker, a managed
   llama-server and the speech process, by exact handle. On Windows a Job
   Object also covers this.

**Failure before the page opens:** the reason goes to the log and to the
user. Windows shows a message box; elsewhere a small `startup-error.html`
page opens in the browser. Nothing fails silently, and no terminal is
needed.

**Other entry points:** `litenotebook <command>` still runs the CLI.
`litenotebook --install-desktop-entry` adds a menu entry for the Linux
tarball. `--internal` runs the POSIX orphan guard inside the frozen app
(`platform_utils.supervise`).

**Windows console:** the core exe is a windowed program (no console). The
worker is a console-subsystem program started with `CREATE_NO_WINDOW`, so
its pipes work and no window appears. llama-server and speech already use
`CREATE_NO_WINDOW`. Diagnostics go to the log file.

## Data, logs, updates

**Data stays in the user's folders, never next to the program:**

- Windows: `%LOCALAPPDATA%\LiteNotebook`
- Linux: `~/.local/share/LiteNotebook` (XDG)

A packaged app never uses "development mode", even if started from a
source tree.

**Logs:** `%LOCALAPPDATA%\LiteNotebook\Logs` or
`$XDG_STATE_HOME/LiteNotebook/logs` (`~/.local/state/…`), in a rotating
`litenotebook.log` (5 × 1 MiB).

- They cover start-up, worker and runtime failures.
- Web access logs are off, because request URLs can contain search text.
- Bearer tokens, `sk-…` keys and `api_key=` values are masked.
- Document text is never logged.
- Settings → Diagnostics → Open logs folder; the Start menu also has a
  logs shortcut.

**Upgrades:** before migrating an existing database to a newer schema,
LiteNotebook copies it with SQLite's online backup to
`<data>/backups/before-schema-<new>-from-<old>-<time>.sqlite3`.

- Migrations never reset data.
- Verified with the built app on real schema 8 (Phase 12), 9 (Phase 13)
  and 10 databases.
- Archive formats 1, 2 and 3 import in the built app, including audio.

**Updating the app:** install the newer version over the old one (manual
for now; no auto-updater). The installer and archives contain only program
files, and the uninstaller removes only those. User data, models, logs and
audio are never deleted; removing them is an explicit, separate user action.

## Windows installer (Inno Setup)

Inno Setup was chosen over WiX (MSI authoring is heavier, and the
per-user + optional-component setup is more work) and NSIS (scripting is
lower level). It is mature, has one script, and compiles on a Windows CI
runner.

**`packaging/windows/litenotebook.iss`:**

- per-user install, no administrator rights;
- Start menu shortcut, plus a "logs folder" shortcut;
- optional desktop shortcut;
- an optional "semantic search runtime" component installed to
  `{app}\embedding-runtime`;
- "Start LiteNotebook" on finish;
- the uninstaller removes program files only.

The installer cannot be built or run on Linux. It has **not** been compiled
or run yet.

## Windows verification gap (all still to do on real Windows; status words in windows.md)

- [ ] Installer installs, Start menu and desktop shortcuts work; uninstall keeps `%LOCALAPPDATA%\LiteNotebook`
- [ ] First launch: no console window; browser opens; setup wizard; hardware detected
- [ ] Data, logs and config under LocalAppData; user names with spaces and non-ASCII characters (`C:\Users\Normal User`, `C:\Users\José`)
- [ ] Second launch opens the running instance; a busy port 5056 leads to another port
- [ ] Optional embedding runtime component: worker starts with no window; PDF upload, indexing, Reader, delete
- [ ] Quit LiteNotebook and Task-Manager kill leave no `litenotebook-embed-worker.exe`, `llama-server.exe` or speech process (Job Object)
- [ ] Backup export, import and restore; schema upgrade with the safety copy
- [ ] Managed llama.cpp (Phase 12) and speech (Phase 14) from the packaged app
- [ ] Long paths where practical; Defender / SmartScreen behaviour on first start

## Licensing (not legal advice)

`docs/THIRD_PARTY_LICENSES.md` is generated by `tools/licenses.py` and also
ships in the app folder. It covers the 53 Python packages plus the native
components, and the optional components that are never bundled.

**PDF engine: PDFium (BSD-3-Clause) via pypdfium2.** PyMuPDF
(AGPL-3.0-or-later or Artifex commercial) was removed in 0.9.2 and is
excluded from the core spec, so it cannot be bundled even from a build
machine that has it (decision: docs/pdf-licensing-decision.md). The licence
texts of PDFium and the libraries compiled into it (FreeType, libjpeg-turbo,
OpenJPEG, libpng, Little CMS, zlib, ICU, Abseil, AGG, fast_float, simdutf,
LLVM libc, pdfium-binaries) are copied from the pypdfium2 wheel into
`licenses/pypdfium2/` in the app folder by `tools/build.py`.

The repository is MIT-licensed, inherited from Open Notebook.

**Other notable components:**

- FAISS (MIT), numpy/scipy (BSD), torch (BSD-style), tokenizers,
  transformers and sentence-transformers (Apache-2.0).
- The PyInstaller bootloader (GPL with a distribution exception).
- libstdc++/libgomp (GPL with the GCC runtime exception).
- Speech and llama.cpp (MIT) are not bundled. Speech is several separately
  licensed parts (sherpa-onnx Apache-2.0, ONNX Runtime MIT, espeak-ng and
  espeak-ng-data GPL-3.0-or-later, Piper voice models per their model cards);
  see `docs/THIRD_PARTY_LICENSES.md` section 3.
