# Security and privacy model (LiteNotebook 0.9.x)

LiteNotebook is a single-user desktop application. Its web UI is served only
to the same computer (127.0.0.1) and its data lives in the user's own
folders. This page says what it is designed to protect against, what it does
not claim, and how the main protections are tested.

Status: **pre-release**. The protections below are implemented and covered by
automated tests on Linux. On Windows they are implemented but not yet
runtime-verified; see [windows.md](windows.md).

## Protects against

| Threat | How | Tests |
|---|---|---|
| **Accidental LAN exposure** | The server binds 127.0.0.1 only. The desktop launcher never uses another address; `serve` refuses non-loopback hosts. Nothing listens on the network and no firewall rule is created. | `test_cli_phase2`, `test_web_security_audit::test_default_bind_is_loopback`, packaged smoke test (LAN address refused) |
| **Other websites in your browser** (CSRF, DNS rebinding) | Only loopback `Host` headers are served. State-changing requests must be same-origin (`Sec-Fetch-Site`, `Origin`, `Referer`); `Origin: null`, other local ports (`127.0.0.1:8080`) and same-site requests are refused. | `test_web_security_audit` (sweeps **every** route), `test_web` |
| **Hostile uploaded documents** | Content sniffing (no trust in names or MIME types), size limits, streamed uploads. Password-protected, damaged and fake PDFs are rejected with a message. Documents are parsed and rendered only: JavaScript, launch, URI and form actions are never executed. Page renders are pixel-capped. PyMuPDF calls are serialised under one lock, because PyMuPDF is not thread-safe. One bad source fails alone. | `test_hostile_pdfs`, `test_sources_web`, `test_pdf` |
| **Prompt injection inside documents** | Document text is fenced as untrusted evidence in every prompt, and the system prompts tell the model to ignore instructions found in it. Quiz and Audio validators drop instruction-like sentences from the support text, including ones wrapped across PDF lines. Answers and turns that rely on them are rejected, as are invented citation labels. | `test_release_security`, `test_study_validate`, `test_audio_validate`, `test_chat_web` |
| **Cross-notebook access by ID tampering** | Every object is looked up through its notebook (`notebook_id` + object id): sources, Reader pages, annotations, chats, citations, notes, links, quizzes, attempts, audio overviews and files. Unknown or foreign ids return 404, and retrieval is scoped per notebook. | `test_release_security`, `test_*_web` scoping tests |
| **Malicious imported backup archives** | Validation runs before any write. It checks entry names (no traversal, absolute paths, symlinks, devices or case collisions), limits sizes, counts and expansion (zip bombs), verifies SHA-256 for every member and the per-row hashes of PDFs and audio, validates and remaps rows, and does not trust embeddings without checking them. A failed import leaves nothing behind; a restore is staged and keeps a safety copy. | `test_archive_security`, `test_archive`, `test_backup_disaster` |
| **Model-generated HTML or script** | All output is escaped (Jinja2 autoescape and Markdown sanitised with nh3). A strict CSP allows no inline script, no `eval` and no third-party origins. Generated JSON is schema-validated. | `test_web`, `test_study_web`, `test_audio_web`, CSP checks |
| **Tampered downloads** | Only HTTPS from catalog hosts. Redirects may only go to the same publishers' CDNs, and the host is checked before the connection is made. Size and SHA-256 are pinned for every runtime, model, voice and engine. MiniLM is pinned to one upstream commit with every file checksummed. Partial files are never used; archives are extracted with the hostile-archive rules; nothing runs before verification. | `test_download_audit`, `test_ai_download`, `test_ai_runtime`, `test_audio_speech` |
| **Secrets and private text in logs** | Logs record what failed (component, error type, process exit, timing), not content. Document text, note bodies, chat prompts and search terms are never logged. A formatter masks API keys, bearer tokens, URL credentials and the home folder (`~`) in every line, tracebacks included. Web access logs and httpx request logs are off. | `test_logging_privacy` |
| **Data-loss on upgrade** | Before a schema upgrade, the database is copied with SQLite's online backup (unique name, written atomically). Each migration runs in a transaction, so a failing migration leaves the last good version. | `test_migration_stress` |

## Does not claim to protect against

- **Malware or another program running as your user account.** It can read
  your data folder, your browser, and the loopback port, like any other file
  or app you use. Encrypting the data folder at rest is out of scope (use OS
  disk encryption).
- **Other user accounts on the same computer.** The loopback port is reachable
  from every account on the machine, and the local UI has **no login**. On a
  shared multi-user computer, another local user could open the UI while
  LiteNotebook is running. File permissions protect the data folder itself.
  *(Known limitation for v1 review.)*
- **A hostile administrator / root** on the machine.
- **A compromised model or runtime you installed yourself** from outside the
  verified catalog (e.g. "use an existing llama-server" or importing your own
  GGUF). LiteNotebook runs what you point it at.
- **A compromised upstream catalog entry.** The catalog pins checksums, so a
  changed file is refused. But if a publisher's original release was
  malicious when it was pinned, the checksum matches. Trusted upstreams are
  listed below.
- **Chat answers that obey a document.** Chat output is free text from the
  model; LiteNotebook fences the evidence and validates citations, but it
  cannot prove a free-text answer ignored a planted instruction. Quiz and
  Audio have validators; Chat relies on the prompt and citations.

## Trusted upstreams (downloads only on request)

| Component | Host | Pinning |
|---|---|---|
| llama.cpp `llama-server` | github.com/ggml-org/llama.cpp releases | release `b11146`, size + SHA-256 per asset |
| Qwen3 GGUF models | huggingface.co/Qwen | file size + SHA-256 |
| MiniLM (all-MiniLM-L6-v2) | huggingface.co/sentence-transformers | commit `1110a243…`, SHA-256 for all 10 files |
| sherpa-onnx engine, Piper voices | github.com/k2-fsa/sherpa-onnx releases | version 1.13.8, size + SHA-256 |
| Embedding runtime (packaged app) | the user's own file, checked against `litenotebook-build.json` | name + size + SHA-256 |

Redirects are allowed only to `*.githubusercontent.com`, `*.huggingface.co`
and `*.hf.co`.

## Deleting data

What each delete removes, and what is intentionally kept:

| Delete | Removed | Intentionally kept elsewhere |
|---|---|---|
| Source | PDF file, pages, chunks, full-text entries, FAISS vectors, render cache, its annotations | Short **excerpts** that other objects cited stay inside those objects, so history keeps its meaning: chat citations (`message_citations.chunk_text`), quiz evidence, audio evidence and note-link excerpts. They are removed when you delete that chat, quiz, overview or note. |
| Conversation | messages, citations, memory references | — |
| Note | note text, its links | — |
| Quiz | questions, options, evidence snapshots, attempts; concept stats recomputed | — |
| Audio overview | transcript, evidence snapshots, WAV files | — |
| Notebook | everything above for the notebook, its folder and index | — |

Since 0.9.1, SQLite's `secure_delete` is on for every connection. FTS5's
`secure-delete` is on for every full-text table (SQLite ≥ 3.42), so deleted
text is overwritten instead of lingering in free pages or index segments
(`test_data_deletion`).

**Limits:**
- Text deleted *before* 0.9.1 may remain in free database pages until the
  database is compacted.
- Older SQLite versions (< 3.42) cannot read the full-text tables after
  secure-delete has been used, until they are rebuilt. LiteNotebook's packages
  bundle a newer SQLite.
- Backups you exported, and restore safety copies
  (`<data>.pre-restore-*`, `<data>/backups/before-schema-*`), are separate
  copies. Delete them yourself.
- The SSD/filesystem may keep old blocks; use OS-level encryption for strong
  guarantees.

## Not in the browser

- No debug mode, no `/docs`, no OpenAPI schema.
- Unexpected errors show a generic page; details go to the local log only.
  Disk-full errors say so (HTTP 507).
- No telemetry and no update checks. LiteNotebook connects to the internet
  only when you start a download (verified by `tools/release_stress.py
  offline`, which runs the app in a network namespace and traces every
  `connect()`).
