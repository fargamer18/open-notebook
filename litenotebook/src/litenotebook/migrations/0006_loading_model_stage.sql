-- Phase 6: MiniLM runs in a child process that starts on demand, so ingestion
-- has a real "loading the embedding model" step. Add it to the stage
-- vocabulary. SQLite cannot alter a CHECK constraint: rebuild `sources` exactly
-- as in 0003 (foreign keys are off while migrating, so nothing cascades), with
-- every column and id preserved.

CREATE TABLE sources_new (
    id                INTEGER PRIMARY KEY,
    notebook_id       INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title             TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path       TEXT NOT NULL,
    mime              TEXT NOT NULL,
    sha256            TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    page_count        INTEGER,
    status            TEXT NOT NULL
                      CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    stage             TEXT NOT NULL DEFAULT 'uploaded'
                      CHECK (stage IN ('uploaded', 'extracting', 'chunking',
                                       'loading_model', 'embedding', 'indexing',
                                       'ready', 'failed')),
    progress_done     INTEGER,
    progress_total    INTEGER,
    ocr               INTEGER NOT NULL DEFAULT 0 CHECK (ocr IN (0, 1)),
    error             TEXT,
    timings           TEXT,
    attempts          INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    started_at        TEXT,
    finished_at       TEXT,
    updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (notebook_id, sha256)
);

INSERT INTO sources_new (id, notebook_id, title, original_filename, stored_path, mime,
                         sha256, size_bytes, page_count, status, stage, progress_done,
                         progress_total, ocr, error, timings, attempts, created_at,
                         started_at, finished_at, updated_at)
SELECT id, notebook_id, title, original_filename, stored_path, mime, sha256,
       size_bytes, page_count, status, stage, progress_done, progress_total, ocr,
       error, timings, attempts, created_at, started_at, finished_at, updated_at
FROM sources;

DROP TABLE sources;
ALTER TABLE sources_new RENAME TO sources;
CREATE INDEX idx_sources_notebook ON sources(notebook_id);
CREATE INDEX idx_sources_status ON sources(status);
