# Release checklist (before any public v1)

0.9.x is a pre-release. Do **not** tag `v1.0` until every blocker is closed.

## Blockers

- [x] **PDF engine licensing decided and applied.** Option C: PyMuPDF
      replaced with PDFium/pypdfium2 in 0.9.2 (Phase 17;
      [pdf-licensing-decision.md](pdf-licensing-decision.md)).
      `THIRD_PARTY_LICENSES.md` regenerated; PDFium notices ship in
      `licenses/pypdfium2/`. Re-check before each release that no package
      contains `pymupdf`/`libmupdf` (the packaged smoke test does).
- [ ] **Windows Built** on Windows (CI `windows-2022` job or Windows 11).
- [ ] **Windows automated tests passed** on Windows: source suite, packaged
      smoke (with and without the runtime), silent install/uninstall.
- [ ] **Windows manually verified**: [windows-test-plan.md](windows-test-plan.md),
      including cross-platform backups.
- [ ] **Linux release artifacts built on Ubuntu 22.04**
      (`packaging/linux/build-in-ubuntu-2204.sh` or CI), with
      `build-report.json` showing `min_glibc` ≤ 2.35.

## Every release build

- [ ] `uv run pytest -q`, `uv run ruff check .`, `uv run mypy --strict src` are clean.
- [ ] The version matches in `src/litenotebook/__init__.py`, `pyproject.toml`
      and `uv.lock`; the build preflight enforces this.
- [ ] `build-report.json`: version, git commit (not `+dirty`), Python,
      PyInstaller, runner, architecture, artifacts with SHA-256.
- [ ] `SHA256SUMS` verifies.
- [ ] `tools/packaged_smoke.py` passes against the built package, with and
      without `--runtime-archive`.
- [ ] `tools/release_stress.py lifecycle DIST --runtime-archive …`: no
      leftovers, no zombies.
- [ ] `tools/release_stress.py offline`: no non-loopback `connect()`.
- [ ] `docs/THIRD_PARTY_LICENSES.md` regenerated (`python tools/licenses.py
      docs/THIRD_PARTY_LICENSES.md`) and reviewed.
- [ ] README status, INSTALL, windows.md and linux.md match reality.
- [ ] No build artifacts tracked by git (`dist/` and `build/` are ignored).

## Screenshots (for the README / release page)

Take these from the release build, using a demo notebook only (never
personal data), at 1366×768 in light and dark mode. Do not redesign the UI
for them.

- [ ] Notebook workspace (sources list with statuses)
- [ ] Reader with a citation highlight and an annotation
- [ ] Chat answer with citations
- [ ] Study → quiz question with feedback, and the result/review page
- [ ] Study → Audio Overview player with chapters and transcript
- [ ] Knowledge graph: Global view, a Local view with the inspector open
- [ ] Setup wizard: hardware step and Local AI settings
