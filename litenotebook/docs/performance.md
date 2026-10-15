# Performance budgets and observed numbers (0.9.3)

These are rough release budgets, not microbenchmarks. The numbers below were
observed on the development machine:

- AMD Ryzen 7 4800H (8 cores / 16 threads), NVMe for the OS and an HDD
  (NTFS) for the source tree;
- Linux, source mode, real MiniLM worker on the CPU, fake local LLM.

They come from `tools/release_stress.py stress`, which builds a large
synthetic notebook in a temporary folder. There are no brittle timing tests
in the suite; the only automated guard is that page **query counts** do not
grow with data size (`tests/test_query_budget.py`).

## Budgets

| Budget | Rule | Observed |
|---|---|---|
| Ordinary server-rendered page | < 100 ms | 4–20 ms median for notebooks, workspace, sources, notes (400), study (4 quizzes / 60 attempts), Reader page, settings, memory graph |
| Long chat page | < 250 ms | 78 ms median / 122 ms p95 with a 120-turn conversation (600 citations) |
| Ordinary navigation starts no AI process | always | ✅ page views, the Reader, keyword search, quiz history and audio playback never start the embedding worker, llama-server or speech (`test_component_states`, `test_reader_never_starts_the_worker`, `test_history_transcript_and_playback_start_nothing`) |
| Fast (keyword) search | interactive (< 100 ms) | 17 ms median over 1,038 chunks + 400 notes + 240 messages |
| Hybrid search, warm | interactive | 23 ms median |
| Hybrid search, cold | explained in the UI | ~23 s the first time (MiniLM worker cold start; PyTorch import). The UI says "may load the embedding model (~15 s)" |
| Reader page image | cached interactive | 8 ms cached, 61 ms first render (1.5×, pixel-capped) |
| Knowledge graph API | interactive | 4–7 ms for a small notebook; 15–22 ms with 1,200 graph nodes / 4,200 links (Global capped at 400 nodes, Local depth 3 at 150); see graph.md |
| Knowledge graph in the browser | 60 fps | 400 nodes: first draw 0.3 s, settled 1.7 s, 16.7 ms median frame during layout; 1,200 nodes: settled 1.9 s, still 60 fps (this machine, headless Chromium) |
| App start → browser | ~1 s | packaged: 0.9–1.1 s to ready, second launch 0.3 s, quit 0.4 s; source mode 1.4–1.7 s, also with the large notebook |
| Ingestion | progress shown | 1,038 chunks (520-page PDF) in 74 s, including the worker's cold start (~14 chunks/s overall) |
| Full backup | seconds | 0.43 s for the large notebook (2.1 MiB archive) |
| Memory (server process) | no unbounded growth | 108 MiB idle → 150–166 MiB after heavy use; six rounds of Reader renders, searches, chats, quiz generation and export/import grew it by 0.4 MiB in total (165.4 → 165.8 MiB): no leak. The worker (~500 MiB) is a separate process that stops after 2 idle minutes |

## Query and DOM sizes (large notebook)

| Page | App SQL statements | HTML | Elements |
|---|---|---|---|
| Notebooks | 6 | 14 KiB | 180 |
| Workspace / sources | 7 | 14 KiB | 255 |
| Chat (120 turns) | 16 | 17 KiB | 273 |
| Notes (400 notes) | 7 | 90 KiB | 1,748 |
| Study | 13 | 12 KiB | 251 |
| Reader page | 7 | 11 KiB | 173 |

**The fast-search page.** It reported ~3,400 statements in the raw SQLite
trace. Almost all of them are SQLite's own FTS5 work: BM25 reads one
document-size row for every matching row. One real N+1 was found and fixed
in 0.9.1: memory search loaded matching notes one query at a time; it now
uses one query per 500. `tests/test_query_budget.py` checks that app-level
statements stay flat from 5 to 80 notes on search, notes, memory, study and
workspace.

**The notes list** is not paginated. 400 notes produce a 90 KiB page, which
is acceptable. Pagination is a possible later improvement, not a release
blocker.

## Final re-check (Phase 19, 0.9.3)

Same machine and method (`tools/release_stress.py stress`, the browser smoke
and the packaged lifecycle run). No regression in the app; one environment
effect is explained below.

| Budget | Phase 16–18 | Phase 19 |
|---|---|---|
| Packaged start → ready | 0.9–1.1 s | 0.88–0.98 s over 8 runs; 0.9 s in a clean Ubuntu 22.04 |
| Source start (large notebook) | 1.4–1.7 s | 2.5–2.6 s (source tree on an NTFS/FUSE hard drive, cold cache); 1.3–1.5 s warm in the browser smoke |
| Ordinary pages (median) | 4–20 ms | 4–17 ms (memory list 17 ms, graph page 5 ms) |
| Long chat (120 turns) | 78 / 122 ms | 72 / 113 ms (median / p95) |
| Keyword search | 17 ms | 20 ms |
| Hybrid search, warm | 23 ms | 31 ms |
| Reader page image | 8 ms cached, 61 ms first | 7 ms cached, 34 ms first |
| Graph API / browser | 4–22 ms; settled 1.7 s | graph page 5 ms; settled 1.6 s; 0–2 ms of browser work per 2 s once settled |
| Full backup (large notebook) | 0.43 s | 0.19 s |
| Ingestion, 520-page PDF (1,038 chunks) | 74 s | 82 s (includes the worker's cold start, see below) |
| Hybrid search, cold (source mode) | ~23 s | 60 s, see below |
| Embedding worker cold start, packaged (NVMe) | — | upload → Ready of a small PDF in 5.1 s, worker start included |
| Server memory over repeated work | +0.4 MiB over six rounds | 136.9 → 137.7 MiB over six rounds (max +0.6 MiB per round, not monotonic): no leak |

**Cold hybrid search in source mode.** The library versions are identical
to Phase 16 (`uv.lock`). On this machine the source checkout and its virtual
environment sit on an NTFS hard drive mounted through FUSE; with a cold page
cache `import sentence_transformers` alone took 33 s and a full MiniLM load
25–32 s. The packaged embedding runtime on an NVMe drive starts in a few
seconds. The UI's "~15 s" note is a typical value, not a bound: slow disks
take longer.
