# LiteNotebook on Windows

## Status (0.9.2): **Prepared** — not Built, not tested, not verified

LiteNotebook uses four separate status words for Windows. They are never
collapsed into one "supported" checkbox.

| Status | Meaning | Windows today |
|---|---|---|
| **Prepared** | Code paths, installer script and CI job exist; Windows behaviour is simulated in tests on Linux | ✅ yes |
| **Built** | The Windows package was produced by the release build on Windows (CI `windows-2022` or a Windows 11 machine) | ❌ not yet: no Windows runner was available in Phase 16, and this repository's GitHub Actions cannot run the LiteNotebook workflow until it moves to its own repository |
| **Automated tests passed** | On real Windows: the source test suite, the packaged smoke test (with and without the embedding runtime) and silent install / uninstall all pass (`ci/litenotebook-build.yml`) | ❌ not yet |
| **Manually verified** | A person ran [windows-test-plan.md](windows-test-plan.md) on a Windows desktop (browser, console windows, SmartScreen, Start menu, uninstall) | ❌ not yet |

**Windows is a v1 release blocker until all four are ✅.** Simulated
Windows tests on Linux (path rules, `ntpath`, lock and process code with
fakes) count only as **Prepared**.

## What is prepared

| Area | Implementation |
|---|---|
| Installer | Inno Setup, per user (`%LOCALAPPDATA%\Programs\LiteNotebook`), no admin rights, Start menu + optional desktop shortcut, optional "semantic search runtime" component; uninstall removes program files only (`packaging/windows/litenotebook.iss`) |
| No console windows | windowed `LiteNotebook.exe`; worker, llama-server and speech started with `CREATE_NO_WINDOW` |
| Data | `%LOCALAPPDATA%\LiteNotebook` (logs in `…\Logs`); paths with spaces and non-ASCII names are handled with `pathlib` and UTF-8 everywhere |
| One instance per data folder | `msvcrt.locking`; a second launch opens the running one |
| Child processes | Job Object (kill-on-close) for the worker, llama-server and speech: no orphans if the app is killed |
| Files in use | renames/replacements retried on sharing violations; restore switches folders only at start-up |
| Errors before the page opens | native message box (`MessageBoxW`) + log file |

## How Windows gets verified

1. Move `ci/litenotebook-build.yml` to `.github/workflows/` in the
   LiteNotebook repository and run it (or run the same steps on Windows 11:
   `uv sync --frozen --group packaging --group dev`, `uv run pytest`,
   `uv run --group packaging python tools/build.py`, then
   `tools/packaged_smoke.py`). This gives **Built** and **Automated tests passed**.
2. On a Windows desktop, follow [windows-test-plan.md](windows-test-plan.md),
   including:
   - a user profile with a space and a non-ASCII character;
   - a Linux-created backup imported on Windows, and a Windows backup taken
     back to Linux;
   - Task-Manager kill and uninstall.

   This gives **Manually verified**.
3. Record results in the plan's results table, with the build's
   `build-report.json` (git commit, runner, checksums).

## Known unknowns until then

- Microsoft Defender / SmartScreen reputation for an unsigned installer.
  Code signing is not set up.
- Behaviour on Windows on ARM (x64 emulation): not targeted.
- Long paths (> 260 characters) beyond what Python enables by default.
- Real GPU (CUDA) llama.cpp builds on Windows. CPU builds are the safe default.
