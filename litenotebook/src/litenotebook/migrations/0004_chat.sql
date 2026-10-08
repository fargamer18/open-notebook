-- Phase 4: persistent per-notebook chat with immutable citation snapshots.
--
-- Message lifecycle (assistant messages):
--   generating -> complete | failed | interrupted
--   failed/interrupted -> generating (retry reuses the same row)
-- User messages are always 'complete'. Each assistant message answers exactly
-- one user message (reply_to), so a retry can never duplicate either side.

CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY,
    notebook_id INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title       TEXT NOT NULL CHECK (trim(title) <> ''),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_conversations_notebook ON conversations(notebook_id, updated_at);

CREATE TABLE messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL
                    CHECK (status IN ('generating', 'complete', 'failed', 'interrupted')),
    reply_to        INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    model           TEXT,          -- set when the text came from the LLM
    error           TEXT,          -- user-facing; no paths, tracebacks or keys
    meta            TEXT,          -- JSON: timings, finish_reason, retrieval counts
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at    TEXT,
    CHECK ((role = 'user') = (reply_to IS NULL))
);
CREATE INDEX idx_messages_conversation ON messages(conversation_id, id);
CREATE UNIQUE INDEX idx_messages_one_reply ON messages(reply_to) WHERE reply_to IS NOT NULL;
CREATE INDEX idx_messages_status ON messages(status) WHERE status = 'generating';

-- One row per valid [S#] label in an assistant message. The chunk/source ids
-- are identity references (chunk ids are AUTOINCREMENT and never reused); the
-- snapshot columns keep the cited evidence readable after the source is
-- deleted. Only the cited chunk is copied, never the whole document.
CREATE TABLE message_citations (
    id              INTEGER PRIMARY KEY,
    message_id      INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    label           INTEGER NOT NULL CHECK (label >= 1),
    ordinal         INTEGER NOT NULL CHECK (ordinal >= 0),  -- order of first appearance
    chunk_id        INTEGER NOT NULL,
    source_id       INTEGER NOT NULL,
    source_filename TEXT NOT NULL,
    page_no         INTEGER NOT NULL,
    chunk_ord       INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    UNIQUE (message_id, label)
);
CREATE INDEX idx_citations_message ON message_citations(message_id);
