# Local AI (Phase 12)

LiteNotebook's AI is a set of optional components, each installed, tested
and switched off on its own:

| Component | What | Used by | Default |
|---|---|---|---|
| Generation | llama.cpp `llama-server` + a GGUF model, or any OpenAI-compatible server | Chat | not set up (the pre-Phase-12 LLM settings keep working) |
| Embeddings | all-MiniLM-L6-v2 in its own worker process | hybrid search, chat retrieval | detected if already in the Hugging Face cache |
| Speech | *reserved* — a future optional text-to-speech component | (audio overviews, later) | — |

## Where things come from

* `src/litenotebook/ai/catalog/*.json` — pinned, versioned catalogs:
  llama.cpp **b11146** (Linux/Windows, CPU/CUDA 12 assets with SHA-256 and
  sizes), three Qwen3 GGUFs at a fixed Hugging Face revision, MiniLM at a
  fixed revision. Generated from the GitHub / Hugging Face APIs, never typed
  by hand. Only HTTPS on `github.com` / `huggingface.co` (and their CDNs as
  redirect targets) is accepted.
* Downloads (`ai/download.py`): disk-space preflight, `.part` file, HTTP
  Range resume (a 206 for the wrong offset restarts), retries with backoff,
  size + SHA-256 check, atomic rename. Never during normal use — only from
  the wizard, Settings → Local AI or `litenotebook ai … install`.
* Runtime archives (`ai/extract.py`) get the backup-archive defences: no
  `..`/absolute/drive/UNC/backslash/reserved names, no duplicates or case
  collisions, size and count limits, links materialised as copies of files
  inside the archive, devices/FIFOs rejected, staging folder swapped in
  atomically.

## Choosing settings

`ai/hardware.py` + `platform_utils/gpu.py` measure CPU, RAM, free disk and
GPUs (NVML for NVIDIA → *confirmed* VRAM; sysfs / registry → *confirmed*,
*estimated* or *unknown*). `ai/planner.py` sizes weights + KV cache
(`2 × layers × kv_heads × head_dim × context × bytes`) + overhead against
usable memory (VRAM − max(10 %, 384 MiB); RAM − 3 GiB − 1 GiB for MiniLM) and
explains every recommendation. GPU names are never used.

Every start (Chat autostart, Settings → Start, Test) re-plans for the memory
that is free **now** — another program may hold the GPU — and on an
out-of-memory exit walks the fallback ladder: smaller context → part of the
model on the CPU → CPU only → most compatible KV cache. What worked is saved
with a note.

## Processes

`ai/runtime_manager.py` starts only the configured executable (inside the
runtime folder for a managed install; a validated `llama-server[.exe]`
otherwise), with an argument list, on 127.0.0.1, through
`platform_utils.processes` (process group / Job Object). A busy port is left
alone and a free one is used. CPU plans hide GPUs (`CUDA_VISIBLE_DEVICES=""`).
Stop, app shutdown and failures stop only that process. If LiteNotebook dies
without shutting down (killed, crashed, terminal closed), the child still
goes: on Linux a tiny guard (`platform_utils/supervise.py`) stops it when the
pipe from LiteNotebook closes; on Windows the Job Object does. Nothing
starts on a page view.

## Interfaces

* Web: `/setup` (wizard: welcome → hardware → embeddings → runtime → model →
  test → ready; every step skippable, resumable), `/settings/ai`, background
  jobs at `/ai/jobs/<id>` (polled; stage changes announced to screen
  readers).
* CLI: `litenotebook ai status | hardware | test | benchmark`,
  `ai runtime list | install | use | test`,
  `ai model list | install | import [--copy] | select | remove`,
  `ai embeddings status | install`. Same `LocalAI` service as the web UI.
* State: `ai.*` JSON rows in the `settings` table (no schema change).

## Release status

Linux: verified on real hardware (see the Phase 12 report). Windows:
implemented and simulated only — see `windows-test-plan.md`.


## Speech (Phase 14, optional)

The third component speaks Audio Overview transcripts with a local voice.
It runs Piper voices on the sherpa-onnx engine and is an optional, explicit
user download: it is installed only from Settings → Local AI → Speech (or
via the "Set up speech" link on an Audio Overview), and nothing is
downloaded during normal use.

Licensing at a glance (not legal advice):

- sherpa-onnx itself is Apache-2.0.
- The current Piper-compatible speech path also uses espeak-ng and its data
  (GPL-3.0), which carry additional GPL-related licensing considerations. The
  complete speech stack is therefore not "Apache-only".
- LiteNotebook does not bundle the speech engine, espeak-ng data or voices in
  its application or installer; the user downloads them from their
  publishers.

See `docs/audio-overviews.md` for the engine choice, voices and licensing.
