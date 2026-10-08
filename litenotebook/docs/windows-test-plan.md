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

## Results

| Date | Windows build | Python | Tester | Result / notes |
|---|---|---|---|---|
| | | | | not yet run |
