# Backup, export, import and restore

Everything is in **Settings → Backup & Restore** (`/backup`) and on each
notebook's export page. Archives are plain ZIP files with JSON tables and
the original PDFs/WAVs. They are portable between Linux and Windows.

| Action | What it does |
|---|---|
| **Export notebook** | one notebook → `.zip`: its sources (original PDFs), pages, chunks with embeddings, chats with citations, notes and links, annotations, reading position, memory, quizzes and attempts, audio transcripts and WAV files |
| **Full backup** | every notebook plus the portable settings (LLM endpoint, model name, top-k). Never included: API keys, absolute paths, caches, logs, downloaded models and runtimes |
| **Import** | adds the notebook(s) from an export or backup under a free name; nothing existing changes |
| **Verify** | checks an archive completely without importing it |
| **Restore** | replaces *all* data with a full backup, at the next start, keeping the current data as a safety copy |

## Formats

| `export_format_version` | Written by | Imports in 0.9.1 |
|---|---|---|
| 1 | Phase 10–12 | ✅ |
| 2 | Phase 13 (adds quizzes) | ✅ |
| 3 | Phase 14+ (adds audio overviews + WAV) | ✅ (current) |
| > 3 | a newer LiteNotebook | refused with "made by a newer version" |

Tested with committed fixtures (`tests/fixtures/phase12_notebook_v1.zip`,
`phase13_notebook_v2.zip`, `phase14_notebook_v3.zip`), in source mode and in
the packaged app.

## Safety rules

- **Validate before any write.** Every member's SHA-256, row structure,
  references, sizes and counts are checked first. Hostile names, symlinks,
  zip bombs and encrypted entries are refused.
- **All or nothing.** A failure while copying or writing (including a full
  disk or Ctrl+C) leaves no partial notebook: rows roll back and the files
  copied so far are removed.
- **Restore is staged.** The backup is imported into `<data>.restore-staging`
  and integrity-checked. At the next start, `<data>` is renamed to
  `<data>.pre-restore-<time>` and the staging folder is renamed into place.
  An interrupted switch is completed or undone at the following start.
  Existing safety folders are never overwritten (a numeric suffix is added).
- **Upgrades keep a copy.** Before a database schema upgrade, LiteNotebook
  writes `<data>/backups/before-schema-<new>-from-<old>-<time>.sqlite3`.

Disaster cases covered by `tests/test_backup_disaster.py`:

- a corrupt manifest, PDF, WAV or embedding;
- consistent tampering, where the checksums were recomputed;
- a truncated ZIP;
- a future format;
- a disk filling up during export, import or restore staging;
- an export destination that disappears (a USB stick pulled out);
- an import interrupted by Ctrl+C;
- a restore interrupted between its two renames, and at the very end;
- safety-folder name collisions;
- a full current-format backup restored into a brand-new folder.

## Recovering by hand

| Situation | What to do |
|---|---|
| A restore went wrong | Quit LiteNotebook, rename `<data>` aside, rename the newest `<data>.pre-restore-*` back to `<data>`, then start LiteNotebook |
| An upgrade went wrong | Quit; replace `<data>/litenotebook.sqlite3` with the matching `<data>/backups/before-schema-*.sqlite3` and use the previous LiteNotebook version |
| You no longer need the copies | Delete the `.pre-restore-*` folders and the `backups/` files yourself; LiteNotebook never deletes them |

Exported archives and safety copies are full copies of your data. Store and
delete them accordingly (see [security.md](security.md#deleting-data)).
