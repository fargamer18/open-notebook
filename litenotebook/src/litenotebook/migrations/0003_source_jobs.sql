-- Phase 3: source ingestion lifecycle for the web app.
--
-- The Phase 1 status vocabulary (queued/extracting/embedding/ready/error) is
-- replaced by pending/processing/ready/failed; the finer-grained progress now
-- lives in `stage`. SQLite cannot alter a CHECK constraint, so `sources` is
-- rebuilt (the migration runner disables foreign keys while migrating, so the
-- DROP below does not cascade into pages/chunks). Ids are preserved.
--
-- Lifecycle: pending (stored, queued) -> processing (extracting .. indexing)
-- -> ready | failed. Pages/chunks are only ever written in the final
-- transaction that also sets `ready`, so a non-ready source has none.

CREATE TABLE sources_new (
    id                INTEGER PRIMARY KEY,
    notebook_id       INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title             TEXT NOT NULL,
    original_filename TEXT NOT NULL,        -- display only, never used as a path
    stored_path       TEXT NOT NULL,        -- relative to the data directory
    mime              TEXT NOT NULL,
    sha256            TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    page_count        INTEGER,
    status            TEXT NOT NULL
                      CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    stage             TEXT NOT NULL DEFAULT 'uploaded'
                      CHECK (stage IN ('uploaded', 'extracting', 'chunking', 'embedding',
                                       'indexing', 'ready', 'failed')),
    progress_done     INTEGER,              -- pages extracted / chunks embedded
    progress_total    INTEGER,
    ocr               INTEGER NOT NULL DEFAULT 0 CHECK (ocr IN (0, 1)),
    error             TEXT,                 -- user-facing message, no paths/tracebacks
    timings           TEXT,                 -- JSON: seconds per ingestion step
    attempts          INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at        TEXT,
    finished_at       TEXT,
    updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (notebook_id, sha256)
);

INSERT INTO sources_new (id, notebook_id, title, original_filename, stored_path, mime,
                         sha256, size_bytes, page_count, status, stage, error,
                         attempts, created_at, finished_at, updated_at)
SELECT id, notebook_id, title, original_filename, stored_path, mime, sha256, size_bytes,
       page_count,
       CASE status WHEN 'ready' THEN 'ready'
                   WHEN 'error' THEN 'failed'
                   WHEN 'queued' THEN 'pending'
                   ELSE 'processing' END,
       CASE status WHEN 'ready' THEN 'ready'
                   WHEN 'error' THEN 'failed'
                   WHEN 'queued' THEN 'uploaded'
                   ELSE 'extracting' END,
       error,
       CASE status WHEN 'queued' THEN 0 ELSE 1 END,
       created_at,
       CASE WHEN status IN ('ready', 'error') THEN created_at END,
       created_at
FROM sources;

DROP TABLE sources;
ALTER TABLE sources_new RENAME TO sources;
CREATE INDEX idx_sources_notebook ON sources(notebook_id);
CREATE INDEX idx_sources_status ON sources(status);
