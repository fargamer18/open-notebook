# Release checklist (v1.0)

0.9.x is a pre-release. Do **not** tag `v1.0` until every **BLOCKER** below
is closed and the remaining **WARNING**s are accepted.

Every item has exactly one status:

- **PASS**: done and checked, with the evidence named.
- **WARNING**: works, with a limitation that must be accepted or documented.
- **BLOCKER**: prevents v1.0.
- **NOT TESTED**: not done; says why.

Last full run: Phase 19 (0.9.3, 2026-10-15), Linux release build from
commit `b63cc34` (clean tree) on Ubuntu 22.04.

## Windows

| Item | Status | Evidence / reason |
|---|---|---|
| Prepared (code, installer script, CI job) | PASS | `packaging/windows/litenotebook.iss`, `ci/litenotebook-build.yml`, Windows code paths simulated in the test suite |
| Built on Windows (PyInstaller + Inno Setup) | **BLOCKER** | No Windows machine, VM or Windows runner was available. The CI job only runs once the workflow is in a repository whose GitHub Actions run it, which needs a push (not authorised). |
| Automated tests passed on Windows (suite, packaged smoke ± runtime, browser smoke, silent install/uninstall) | **BLOCKER** | Same: never run on Windows |
| Manually verified on a Windows desktop ([windows-test-plan.md](windows-test-plan.md)) | **BLOCKER** | Not run. Includes data paths with spaces/Unicode, second instance, PDF delete under Windows file locking, embedding runtime, Local AI, quiz, audio, graph, backup, Quit, Task-Manager kill, uninstall/reinstall |
| Linux → Windows and Windows → Linux backup interop | **BLOCKER** | Needs Windows. Linux side: formats 1–3 import in the packaged smoke test |
| SmartScreen / Defender behaviour of the unsigned installer | NOT TESTED | Needs Windows; see "Signing" below |
| Screen reader (Narrator or NVDA) | NOT TESTED | Needs Windows (no Orca run on Linux either) |

## Linux release build (Ubuntu 22.04 baseline)

| Item | Status | Evidence |
|---|---|---|
| Built on Ubuntu 22.04 from a clean commit | PASS | `packaging/linux/build-in-ubuntu-2204.sh`; `build-report.json`: version 0.9.3, commit `b63cc34…` (not `+dirty`), Ubuntu 22.04.5, Python 3.12.14, PyInstaller 6.22.3 |
| glibc requirement | PASS | `min_glibc` 2.35 (app and runtime), GLIBCXX ≤ 3.4.30 |
| tar.gz, AppImage, embedding runtime, `SHA256SUMS` | PASS | 71,405,486 / 66,660,856 / 329,668,622 bytes; `sha256sum -c SHA256SUMS` OK |
| PyMuPDF absent, PDFium present | PASS | no `*mupdf*`/`fitz*` file; `libpdfium.so` + `licenses/pypdfium2/`; packaged smoke checks both |
| Graph and web assets present | PASS | `graph.js`, `htmx.min.js` + `htmx.LICENSE` in the package |
| No models, user data, secrets or dev paths | PASS | no `.gguf/.safetensors/.onnx/.pt`, no database or notebook files, no torch in the core app; no `/home/…` or source-tree path in any text or binary file or in the build report |
| Fresh Ubuntu 22.04: packaged smoke, core only | PASS | 46/46 in a pristine `ubuntu:22.04` container (glibc 2.35) |
| Fresh Ubuntu 22.04: packaged smoke with the runtime | PASS | 52/52: runtime install, ingestion, FAISS, semantic search, Reader, worker idle exit |
| Fresh Ubuntu 22.04: real browser | PASS | `tools/browser_smoke.py` 38/38 with the app inside the container (host browser over loopback), ending with Quit |
| AppImage with FUSE 2 | PASS | 38/38 browser smoke on Arch (`libfuse.so.2`), Quit leaves no mount or process |
| AppImage without FUSE 2 | WARNING | A direct run fails ("No suitable fusermount binary"). `--appimage-extract-and-run` works (Ubuntu 22.04 container, data path with spaces and "José"). Ubuntu 22.04+ does not install `libfuse2` by default. Documented in INSTALL.md / linux.md |
| Distributions other than Ubuntu 22.04 and Arch | NOT TESTED | Expected from the glibc 2.35 baseline; no claim of universal Linux support |
| Linux arm64 | NOT TESTED | Not built |

## Every release build

| Item | Status | Evidence |
|---|---|---|
| `pytest`, `ruff check`, `ruff format --check`, `mypy --strict src` | PASS | see the Phase 19 report |
| Version matches in `__init__.py`, `pyproject.toml`, `uv.lock` | PASS | build preflight |
| Packaged smoke test (± `--runtime-archive`) | PASS | Ubuntu 22.04 container, see above |
| Browser smoke (`tools/browser_smoke.py`), source and package | PASS | 37/37 source, 38/38 package (tar.gz, AppImage, Ubuntu 22.04) |
| Lifecycle (`tools/release_stress.py lifecycle`) | PASS | 8 start → action → quit cycles (ready 0.88–0.98 s), port collision, worker SIGKILL respawns, app SIGKILL reaps the worker, restart after SIGKILL recovers the lock and data; no leftovers, no zombies |
| Offline (`tools/release_stress.py offline`) | PASS | 26 pages/actions incl. stored chats, quizzes, audio transcript, Global/Local graph, local-model chat, export; 0 non-loopback `connect()` |
| `THIRD_PARTY_LICENSES.md` matches the environment | PASS | regenerated with `tools/licenses.py`: no difference |
| Release built from the commit that is tagged | NOT TESTED | Do at tag time: the Phase 19 changes (tools, tests, docs) must be committed and the build repeated from that commit |

## Security and privacy

| Item | Status | Evidence |
|---|---|---|
| Loopback only | PASS | packaged app listens on `127.0.0.1` only; not reachable on the LAN address |
| Host / Origin / CSRF | PASS | foreign `Host` → 400, cross-origin POST → 403 (live probes on the package); `test_web_security_audit` sweeps every route |
| POST without any Origin/Referer/Sec-Fetch-Site | PASS (by design) | treated as a local non-browser client (`web/security.py`); browsers always send one of them cross-site |
| Debug off, no docs/schema, no tracebacks | PASS | `/docs`, `/openapi.json`, `/redoc` → 404; error pages and logs carry no traceback |
| Path traversal, malformed ids | PASS | static traversal → 404; non-integer/huge ids → 422/404; graph ids validated per notebook |
| Notebook isolation | PASS | `test_release_security`, `test_graph_view` (cross-notebook id tampering) |
| XSS payloads, malicious file names | PASS | `test_web`, `test_graph_view::test_untrusted_labels_stay_text`, `test_hostile_pdfs` |
| Hostile PDF suite | PASS | `test_hostile_pdfs` (actions, JavaScript, encryption, truncation, huge pages, fake PDFs) |
| Prompt injection | PASS | `test_release_security` (chat fencing; quiz/audio validation) |
| Download security | PASS | every catalog asset HTTPS, publisher host, size and SHA-256 pinned; MiniLM pinned to one commit; redirects to other hosts refused before connecting (`test_download_audit`) |
| Speech voice archives pinned by version tag | WARNING | they come from sherpa-onnx's rolling `tts-models` release tag. Integrity is still enforced (size + SHA-256, fail-closed), but a re-upload by the publisher would make the install fail until the catalog is updated |
| Log redaction | PASS | home path, credentials, tokens masked (`test_logging_privacy`); no search, note or document text logged in normal use; live run: 0 home-path matches |
| Secure delete | PASS | `test_data_deletion` (secure_delete on, files/vectors/cache removed) |
| Graph payloads carry no document text | PASS | `test_payloads_do_not_carry_document_text` |
| Low disk / permissions | PASS | upload and backup with a full disk, unwritable data folder, unwritable page cache, unwritable log folder (`test_resource_failures`) |

## Data

| Item | Status | Evidence |
|---|---|---|
| Migrations 1–9 → 10 | PASS | `test_migration_stress` (every schema, real older databases, rollback, safety copies, newer-DB refusal, WAL) |
| A real schema-8 database (copy) | PASS | a copy of the real Maths database migrated 8 → 10 with a safety copy; integrity ok, no FK violations, 3 sources / 133 chunks / 1 conversation intact. The original was not opened by the app |
| Backup disaster suite | PASS | `test_backup_disaster` (corruption, tampering, truncation, disk full, interrupted import/restore) |
| Linux export → Linux import, formats 1–3 incl. audio | PASS | packaged smoke test |

## Product checks (Linux)

| Item | Status | Evidence |
|---|---|---|
| Reader: render, zoom, keyboard paging, citation highlight, annotation | PASS | browser smoke |
| Chat citation → evidence → highlighted page | PASS | browser smoke |
| Quiz: take (incl. keyboard), score, review evidence → highlight, retry mistakes, history | PASS | browser smoke |
| Audio transcript citation → Reader | PASS | browser smoke (transcript only) |
| Speech synthesis with the real engine | NOT TESTED | covered with a fake engine in `test_audio_*` (synthesis, cancel, cleanup, WAV playback with Range requests); the real sherpa-onnx engine is not installed on the test machine and was not downloaded in Phase 19 |
| Local AI with a real model | NOT TESTED | no GGUF model is present on the test machine and none was downloaded; fake OpenAI-compatible model in the offline/stress runs; managed llama-server lifecycle (start, health, stop, restart, no orphan after kill) covered with a fake server in `test_ai_runtime`. No GPU claim is made |
| Global/Local graph, inspector navigation, List view, keyboard, reduced motion, idle after settling | PASS | browser smoke and the Phase 18 suite |
| Notes, keyword search, memory | PASS | browser smoke |
| Hybrid search, packaged | PASS | upload → Ready in 5.1 s including the worker's cold start (NVMe) |
| Hybrid search cold start from a source checkout on slow storage | WARNING | ~60 s on this machine (venv on an NTFS/FUSE hard drive; `import sentence_transformers` alone 33 s). The UI's "~15 s" estimate is optimistic there |
| Graph pinch-zoom on touch screens | WARNING | not supported (wheel, buttons and keys zoom) |
| Accessibility: skip link, dialogs, keyboard for Reader/Quiz/Graph, reduced motion | PASS | browser smoke |

## Signing

| Item | Status | Note |
|---|---|---|
| Windows installer code signing | WARNING | Unsigned. Expect a SmartScreen "Windows protected your PC" warning until signed or reputation builds (not observed: needs Windows). Recommendation: sign with an OV/EV certificate (or Azure Trusted Signing) before wide distribution |
| Linux artifacts | PASS (checksums) | `SHA256SUMS`; no GPG signature |

## Screenshots (for the README / release page)

Take them from the release build at 1366×768, using a demo notebook only
(never personal data). The app is dark-only. Do not redesign the UI for them.

- [ ] Notebook home (notebooks list)
- [ ] Workspace with the sources list and statuses
- [ ] Reader with a citation highlight and an annotation
- [ ] Chat answer with citations and the evidence panel
- [ ] Notes list and a search results page
- [ ] Quiz question with feedback, and the result page
- [ ] Audio Overview with chapters and transcript
- [ ] Knowledge graph: Global view
- [ ] Knowledge graph: Local view with the inspector open
- [ ] Settings → Local AI and the setup wizard's hardware step
