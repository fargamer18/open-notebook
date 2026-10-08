"""Portable LiteNotebook archives: notebook export/import and full backups.

Format (version 1) — a ZIP file::

    manifest.json                       format, versions, counts, checksums hash
    checksums.json                      SHA-256 of every other entry
    settings.json                       (full backups only) non-secret settings
    notebooks/<n>/notebook.json         notebook metadata
    notebooks/<n>/tables/<table>.jsonl  one JSON object per row
    notebooks/<n>/sources/<s>.pdf       the original PDF bytes

``<n>`` and ``<s>`` are the exporting machine's ids, used only to connect
rows inside the archive; importing always assigns new local ids. Entry names
are portable forward-slash paths made of application-generated parts, and
original file names exist only as metadata inside the JSON.

Authoritative data is exported (rows, PDFs, chunk embeddings). Derived data
is not: FAISS indexes are rebuilt from the embeddings, the memory graph is
re-derived from conversations, outcomes and notes, FTS tables are filled by
the normal triggers, and rendered page images are re-created lazily.
"""
