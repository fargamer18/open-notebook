# LiteNotebook on Linux

## Status (0.9.3)

| Status | Linux x86_64 |
|---|---|
| Prepared | ✅ |
| Built | ✅ on Ubuntu 22.04 (container build, `packaging/linux/build-in-ubuntu-2204.sh`) and on the Arch development machine |
| Automated tests passed | ✅ source tests (Arch); packaged smoke tests (± embedding runtime) inside a clean Ubuntu 22.04 container; `tools/browser_smoke.py` against the tarball, the AppImage (FUSE 2, Arch) and the app running inside Ubuntu 22.04; lifecycle and offline runs (Phase 19) |
| Manually verified | ✅ on the development machine (Arch Linux, KDE, Firefox/Chromium) |

Linux arm64 is not built.

## Which build to use

| Build | Built on | Needs | Use |
|---|---|---|---|
| **Release** | Ubuntu 22.04 (container script or the CI job) | glibc ≥ 2.35, GLIBCXX ≥ 3.4.30 | anything you give to others |
| Development | your own machine (`tools/build.py`) | that machine's glibc (Arch: 2.38+) | local testing only |

`build-report.json` records the minimum glibc an artifact needs
(`glibc.app.min_glibc`, measured from the ELF symbol versions of every
bundled library) and the files that need it.

Expected to work on Ubuntu 22.04+, Debian 12+, Fedora 36+, openSUSE Leap
15.5+ and current Arch-based distributions. The minimum glibc is measured;
other distributions are not individually tested. **No claim of universal
Linux compatibility is made.**

## AppImage vs tarball

- **AppImage:** one file, mounted with FUSE 2 when run. Ubuntu 22.04 and
  newer do not install `libfuse2` by default; without it a direct start fails
  with "No suitable fusermount binary found". Install `libfuse2`
  (`libfuse2t64` on 24.04), run it with `--appimage-extract-and-run` (verified
  on a clean Ubuntu 22.04), or use the tarball.
- **Tarball:** a plain folder. Add it to the application menu with
  `./litenotebook --install-desktop-entry` (per user, in
  `~/.local/share/applications`).

## Folders

| | Path |
|---|---|
| Data | `$XDG_DATA_HOME/LiteNotebook` (`~/.local/share/LiteNotebook`) |
| Logs | `$XDG_STATE_HOME/LiteNotebook/logs` (`~/.local/state/LiteNotebook/logs`) |
| Config / cache | `$XDG_CONFIG_HOME/LiteNotebook`, `$XDG_CACHE_HOME/LiteNotebook` |
| Embedding runtime (installed from the app) | `<data>/runtime/embedding/<version>/` |

`LITENOTEBOOK_DATA_DIR` or `--data-dir` choose another data folder.

## Processes

LiteNotebook is one process serving `127.0.0.1`. When needed it starts:

- the embedding worker, which stops after 2 idle minutes;
- llama-server, if LiteNotebook manages it;
- the speech engine, while synthesising.

Each child gets a small orphan guard (`--internal
litenotebook.platform_utils.supervise`), which stops it if LiteNotebook is
killed. Verified with `tools/release_stress.py lifecycle` (SIGKILL of the
app and of the worker, repeated start/quit, port collision; no orphans and
no zombies).

## Optional system tools

- `tesseract` (with the `eng` language) enables OCR of scanned PDF pages. It
  is not bundled.
- GPU acceleration for generation uses the NVIDIA driver through
  llama.cpp's CUDA build. The embedding worker always runs on the CPU.
