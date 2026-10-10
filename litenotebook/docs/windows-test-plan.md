# Windows 11 validation checklist

Phase 11 made the code portable and simulated Windows rules on Linux. **None of
the items below has been run on Windows yet.** Mark each one only after doing it
on a real Windows 11 machine (Phase 13 / release testing). Record the Windows
build, Python version and `litenotebook diagnostics platform` output with the
results.

Use a test account or a disposable data folder — never your only copy of data.

## Install / environment (running from source)

- [ ] Install Python 3.12 (python.org installer) and `uv`; `uv sync` succeeds
      (CPU-only torch, faiss-cpu, PyMuPDF wheels install).
- [ ] `uv run python -m litenotebook.cli diagnostics platform` shows Windows,
      architecture, RAM, disk, SQLite with `fts5=True`, and no API key.
- [ ] `uv run pytest -q` passes (model/LLM tests skipped; `posix_only` skipped).
- [ ] `ruff check .` and `mypy --strict src` pass on Windows.
- [ ] Default paths: with `LITENOTEBOOK_MODE=user` and no `LITENOTEBOOK_DATA_DIR`,
      data goes to `%LOCALAPPDATA%\LiteNotebook` (check in Diagnostics).
- [ ] A Windows user name **with spaces** (e.g. `Ana Maria`): start, upload,
      backup, restore all work.
- [ ] A **Unicode** user name (e.g. `Zoë` or `李雷`): same.
- [ ] A data folder path longer than 200 characters works (or fails with a clear
      message — note which).
- [ ] Environment variables set from PowerShell (`$env:...`) and cmd.exe
      (`set ...`) are honoured, including `%USERPROFILE%` inside
      `LITENOTEBOOK_DATA_DIR`.
- [ ] `serve --open` opens the default browser.
- [ ] Later (Phase 12): `llama-server.exe` is discovered on PATH and in a chosen
      folder; Diagnostics shows it.

## Database

- [ ] First start creates the schema (migrations 1–8); WAL files
      (`-wal`, `-shm`) appear and are cleaned up as on Linux.
- [ ] Keyword search (FTS5) returns results.
- [ ] Starting a second `serve` on the same data folder is refused with the
      "already in use" message; read-only `backup verify` still works.
- [ ] Kill the server in Task Manager; the next start works (stale lock file
      ignored).
- [ ] Full backup → restore: staged, then applied on next start; the old folder
      is kept as `….pre-restore-…`; no "file in use" errors.
- [ ] Restore requested while the server runs is applied only after restart.

## Files

- [ ] PDF upload (incl. a file named `CON.pdf`, `nul.pdf`, `a:b.pdf`,
      `trailing dot.pdf.`) is stored as `<id>.pdf`; names show correctly.
- [ ] Delete a source **while its page is open in the reader** in another tab:
      the source is deleted; any page-image leftovers are harmless.
- [ ] Reader: pages render, zoom, annotations save and reload.
- [ ] Notebook delete removes its folder (including read-only files, if any).
- [ ] Interrupted upload / import leaves nothing in `<data>\tmp` after restart.
- [ ] OCR (if tesseract.exe is installed): a scanned page is read, no console
      window flashes up, non-ASCII text is correct.

## Embedding worker

- [ ] First query/ingest spawns `python.exe -m litenotebook.embed.worker`
      with **no console window**; MiniLM loads from the local HF cache.
- [ ] Query embedding works; ingest of a multi-page PDF works.
- [ ] Idle exit: after the idle timeout the worker process disappears.
- [ ] Ctrl+C in the server console stops the server and the worker exits
      (no leftover `python.exe` in Task Manager).
- [ ] Kill the **server** process in Task Manager while the worker runs: the
      worker is terminated too (Job Object) — no orphan `python.exe`.
- [ ] Kill the **worker** in Task Manager: the next embedding starts a new one.

## Backup portability

- [ ] Export a notebook on Windows; verify it on Windows.
- [ ] Import on Windows the Linux-created fixture
      (`python tests/portable_archive.py fixture.zip` on Linux, copy over).
- [ ] Import the Windows-created export on Linux.
- [ ] Re-export after the round trip: same counts, byte-identical PDFs.
- [ ] Archives with `..\`, `C:\`, `\\server\share`, `CON` entries are rejected
      (run `pytest tests/test_archive_security.py` on Windows).

## UI (browser)

- [ ] Every page from the Phase 9 checklist works in Edge and Chrome:
      notebooks, sources, chat (with llama-server.exe running), notes, search,
      memory graph, reader, Backup & Restore, Diagnostics (Copy works).
- [ ] Downloads (export/backup) get a sane file name and open as ZIP files.

## Local AI (Phase 12)

Status: implemented and **simulated only** on Linux (`.exe` names, flat ZIP
layout, Windows paths with spaces, registry GPU detection code path). None of
this is verified until the boxes below are ticked on a real Windows machine.

- [ ] `pytest tests/test_ai_planner.py tests/test_ai_download.py` pass
      (`test_ai_runtime.py` / `test_ai_web.py` fake-server tests are POSIX-only
      where they need a shebang script; note which ran).
- [ ] `litenotebook ai hardware`: CPU, cores, RAM, free disk correct; an NVIDIA
      GPU shows VRAM as *confirmed* (NVML via `nvml.dll`), other GPUs as
      *estimated* (registry) or *unknown* — never guessed; no console window
      flashes.
- [ ] `litenotebook ai runtime list` shows only the `windows-x86_64` builds;
      CUDA is offered only with an NVIDIA driver ≥ the catalog minimum.
- [ ] Install the CPU build from the wizard: progress shows real bytes,
      Cancel works, a second Install resumes the `.part`; the ZIP (flat
      layout, `llama-server.exe` at the root) extracts under
      `%LOCALAPPDATA%\LiteNotebook\runtime\llama.cpp\…`; nothing is left in
      `tmp\downloads`.
- [ ] Install the CUDA build (two ZIPs: the build + `cudart` DLLs) on an
      NVIDIA machine; `ai runtime test` lists `CUDA0`.
- [ ] "Use an existing llama-server": choosing `llama-server.exe` works; any
      other `.exe` (e.g. `notepad.exe`) is refused; a path with spaces works.
- [ ] Import a GGUF from a path with spaces and non-ASCII characters, by
      reference and with "Copy"; Remove deletes only the copy.
- [ ] Test / Benchmark start `llama-server.exe` with no console window
      (`CREATE_NO_WINDOW`), bound to 127.0.0.1; Windows Firewall shows **no**
      prompt (loopback only).
- [ ] Port 8080 already used by another program → LiteNotebook picks another
      port and never stops that program.
- [ ] Stop (Settings), closing LiteNotebook (Ctrl+C) and killing LiteNotebook
      from Task Manager all leave **no** `llama-server.exe` behind (Job Object).
- [ ] Chat autostarts llama-server when Local AI is set up, and explains
      "Local AI is not configured" with a setup link when generation is off.
- [ ] "Install MiniLM" downloads into `%USERPROFILE%\.cache\huggingface`
      only when pressed; normal use stays offline.
- [ ] "Open models folder" opens Explorer at the models folder.
- [ ] Antivirus (Defender) does not quarantine the extracted runtime; note
      the first-start time (SmartScreen / scanning can slow it).

## Study / quizzes (Phase 13)

- [ ] Study tab appears between Notes and Memory; Study home and quiz history
      open with Local AI stopped (no llama-server.exe, no MiniLM worker in
      Task Manager).
- [ ] Generate a 5-question quiz from two of three PDFs; the stages are shown,
      Cancel stops `llama-server.exe` generating (CPU drops), and no second
      `llama-server.exe` is started for quizzes while Chat is used.
- [ ] Take the quiz with the keyboard only (arrow keys, Space, Enter, 1–4);
      Hint, explanation, evidence and "Open in PDF" (Reader jumps to the
      highlighted passage, "Back to quiz" returns).
- [ ] Retry mistakes, result page, history; delete a quiz (confirmation).
- [ ] Export the notebook, import it on the same machine and from the Linux
      archive: quizzes, attempts and evidence survive; a Phase 12 (v1) backup
      still imports.
- [ ] Source file names with spaces / non-ASCII characters appear correctly in
      evidence labels.

## Audio Overviews and speech (Phase 14)

- [ ] Study shows Quizzes and Audio Overviews; no new notebook tab.
- [ ] Generate a ~5 min two-person transcript with Local AI; transcript,
      citations and "Open in PDF" work with speech not installed.
- [ ] Settings → Local AI → Speech: install the engine
      (`sherpa-onnx-v1.13.8-win-x64-shared-MT-Release`). The archive layout
      is marked unverified: confirm `bin\sherpa-onnx-offline-tts.exe` starts
      with the DLLs from `lib\` on PATH, with no console window.
- [ ] Add two voices; "Test voice" plays the sample in the browser.
- [ ] Generate audio: two distinct voices, chapters seek, the transcript
      follows playback; Task Manager shows at most one
      `sherpa-onnx-offline-tts.exe` at a time and none afterwards.
- [ ] Cancel during synthesis: the engine process disappears; the transcript
      stays; no `.partial` / `.tmp-*` left under `notebooks\<id>\audio`.
- [ ] Data folder and title with spaces / non-ASCII characters: audio plays
      and downloads with a valid file name.
- [ ] Delete the overview while the audio was just played (file handles
      closed): the folder is removed.
- [ ] Export / import a notebook with audio; import Phase 12 and Phase 13
      archives.

## Status words

Record each result as one of: **Prepared** (code exists, simulated on
Linux), **Built** (package produced on Windows), **Automated tests passed**
(suite + packaged smoke + silent install/uninstall on Windows), **Manually
verified** (this plan completed by a person). As of 0.9.1, Windows is
**Prepared** only.

## Release verification (Phase 16) — run on Windows 11 or the CI runner

- [ ] `uv sync --frozen --group packaging --group dev`; `uv run pytest -q` passes on Windows
- [ ] `uv run --group packaging python tools/build.py` builds the zip, installer and runtime zip; `SHA256SUMS` verifies
- [ ] `uv run python tools/packaged_smoke.py dist\litenotebook --export-to exports` passes
      (loopback-only check, Unicode/space data path, formats 1–3 import incl. audio playback,
      source delete with no open-file errors, restart persistence, Quit, no leftover process)
- [ ] the same with `--runtime-archive dist\litenotebook\LiteNotebook-embedding-runtime-*.zip`
      (worker starts with no window, embeds, stops after the idle timeout)
- [ ] silent install → `tools/packaged_smoke.py --app-folder "%LOCALAPPDATA%\Programs\LiteNotebook"` → uninstall keeps `%LOCALAPPDATA%\LiteNotebook`
- [ ] Linux-created archive imports on Windows (the smoke test imports the committed fixtures);
      the Windows export in `exports\` imports on Linux (`litenotebook` → Backup → Import)
- [ ] Diagnostics shows "packaged" and Windows; Settings → Local AI and Setup start no download
- [ ] Task Manager: kill LiteNotebook.exe while the worker / llama-server runs → children gone (Job Object)
- [ ] Two LiteNotebook launches → one server, the second opens the same page
- [ ] No console window ever appears (app, worker, llama-server, speech)
- [ ] Existing quiz take/review works and an imported WAV plays in the browser

## Desktop package (Phase 15)

Run the BUILT package (from the Windows CI workflow or `tools/build.py` on
Windows), not the source tree. See the gap list in `docs/packaging.md`.

- [ ] `tools/build.py` succeeds on Windows (PyInstaller + Inno Setup)
- [ ] `python tools/packaged_smoke.py dist\litenotebook` passes (with and
      without `--runtime-archive`)
- [ ] Installer: per-user install, Start menu + optional desktop shortcut,
      "Start LiteNotebook" on finish, uninstall keeps `%LOCALAPPDATA%\LiteNotebook`
- [ ] No console window for LiteNotebook.exe, the worker, llama-server or speech
- [ ] User profile paths with a space and with non-ASCII characters
- [ ] Quit LiteNotebook / Task-Manager kill: no child process left
- [ ] Upgrade a Phase 12–14 data folder: data kept, safety copy in `backups\`

## Results

| Date | Windows build | Python | Tester | Result / notes |
|---|---|---|---|---|
| | | | | not yet run |
