-- Phase 1: notebooks, sources, pages, chunks (+ FTS5).
-- SQLite is the source of truth; FAISS indexes are rebuilt from chunks.embedding.

CREATE TABLE notebooks (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE CHECK (trim(name) <> ''),
    description TEXT NOT NULL DEFAULT '',
    embed_model TEXT NOT NULL,
    embed_dim   INTEGER NOT NULL CHECK (embed_dim > 0),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE sources (
    id                INTEGER PRIMARY KEY,
    notebook_id       INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title             TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_path       TEXT NOT NULL,        -- relative to the data directory
    mime              TEXT NOT NULL,
    sha256            TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    page_count        INTEGER,
    status            TEXT NOT NULL
                      CHECK (status IN ('queued', 'extracting', 'embedding', 'ready', 'error')),
    error             TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (notebook_id, sha256)
);
CREATE INDEX idx_sources_notebook ON sources(notebook_id);

CREATE TABLE pages (
    id        INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    page_no   INTEGER NOT NULL CHECK (page_no >= 1),   -- 1-based PDF page number
    text      TEXT NOT NULL,                          -- chunk offsets index into this
    method    TEXT NOT NULL CHECK (method IN ('text', 'ocr', 'empty')),
    UNIQUE (source_id, page_no)
);

CREATE TABLE chunks (
    -- Also the FAISS vector id. AUTOINCREMENT: ids are never reused, so a
    -- stale vector in FAISS can never masquerade as a newer chunk.
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    notebook_id INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    page_id     INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    page_no     INTEGER NOT NULL CHECK (page_no >= 1),
    ord         INTEGER NOT NULL CHECK (ord >= 0),    -- order within the source
    char_start  INTEGER NOT NULL CHECK (char_start >= 0),
    char_end    INTEGER NOT NULL CHECK (char_end > char_start),
    heading     TEXT,                                 -- Markdown heading path (later)
    text        TEXT NOT NULL,
    n_tokens    INTEGER NOT NULL CHECK (n_tokens BETWEEN 1 AND 256),  -- MiniLM, incl. [CLS]/[SEP]
    embedding   BLOB,                                 -- little-endian float32[embed_dim], L2-normalized
    UNIQUE (source_id, ord)
);
CREATE INDEX idx_chunks_notebook ON chunks(notebook_id);
CREATE INDEX idx_chunks_page ON chunks(page_id);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content = 'chunks',
    content_rowid = 'id',
    tokenize = 'porter unicode61'
);

CREATE TRIGGER chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER chunks_fts_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
