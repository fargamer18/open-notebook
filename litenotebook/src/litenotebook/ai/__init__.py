"""Local AI setup and management (Phase 12).

LiteNotebook's AI is a set of optional *components*, each independently
installed, tested and switched off:

* ``generation`` — a llama.cpp server (installed by LiteNotebook, a
  llama-server the user already has, or any OpenAI-compatible server) plus a
  GGUF model. Used by Chat today and, later, by quizzes and podcast scripts.
* ``embeddings`` — MiniLM for semantic search, run in its own process.
* ``speech`` — reserved for a future optional text-to-speech component; the
  registry (``components``) is built so it can be added without redesign.

Rules that hold everywhere in this package:

* Nothing is ever downloaded during normal use. Downloads happen only from
  the setup wizard, Settings → Local AI, or an explicit CLI command, and the
  user sees what, how large, from where and under which license first.
* Everything downloaded comes from the pinned, versioned catalogs in
  ``ai/catalog/*.json`` and is verified by SHA-256 before use; runtime
  archives get the same hostile-archive defences as notebook backups.
* Processes are started through ``platform_utils.processes`` (exact handles,
  no shell, no killing by name), on 127.0.0.1 only.
* The web parent never imports torch / sentence-transformers.
"""
