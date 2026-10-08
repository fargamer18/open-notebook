-- Phase 5: deterministic graph memory over past conversations.
--
-- Authoritative data: conversations, messages, message_citations (Phase 4)
-- and memory_outcomes (user feedback, below). The graph (memory_nodes,
-- memory_edges) is *derived* from them by deterministic code (no LLM) and can
-- be rebuilt at any time; rebuilding never touches outcomes or corrections.
--
-- Node identity is a deterministic `key` (e.g. 'answer:42', 'concept:matrix'),
-- unique per notebook, so syncing is an upsert and ids stay stable.
-- Graph rows of a conversation disappear with the conversation (CASCADE);
-- they never depend on a PDF source still existing (evidence nodes carry the
-- citation snapshot, and source/chunk ids are identity values, not FKs).

CREATE TABLE memory_outcomes (
    id          INTEGER PRIMARY KEY,
    notebook_id INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    message_id  INTEGER NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
    outcome     TEXT NOT NULL CHECK (outcome IN ('useful', 'dead_end', 'corrected')),
    correction  TEXT CHECK (correction IS NULL OR length(correction) <= 2000),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK ((outcome = 'corrected') = (correction IS NOT NULL AND trim(correction) <> ''))
);
CREATE INDEX idx_outcomes_notebook ON memory_outcomes(notebook_id, outcome);

CREATE TABLE memory_nodes (
    id              INTEGER PRIMARY KEY,
    notebook_id     INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    key             TEXT NOT NULL,
    node_type       TEXT NOT NULL CHECK (node_type IN (
                        'conversation', 'question', 'answer', 'concept',
                        'source_evidence', 'correction')),
    label           TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    provenance      TEXT NOT NULL CHECK (provenance IN (
                        'user_statement', 'assistant_response', 'source_backed',
                        'user_correction', 'derived')),
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    message_id      INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    source_id       INTEGER,   -- identity only (the source may be deleted later)
    chunk_id        INTEGER,   -- identity only
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (notebook_id, key)
);
CREATE INDEX idx_memory_nodes_type ON memory_nodes(notebook_id, node_type);
CREATE INDEX idx_memory_nodes_message ON memory_nodes(message_id);
CREATE INDEX idx_memory_nodes_conversation ON memory_nodes(conversation_id);

CREATE TABLE memory_edges (
    id           INTEGER PRIMARY KEY,
    notebook_id  INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    from_node_id INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    to_node_id   INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    relation     TEXT NOT NULL CHECK (relation IN (
                     'contains', 'answered_by', 'followed_by', 'mentions',
                     'supported_by', 'corrects')),
    provenance   TEXT NOT NULL,   -- which authoritative row produced it, e.g. 'message:42'
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (from_node_id <> to_node_id),
    UNIQUE (from_node_id, to_node_id, relation)
);
CREATE INDEX idx_memory_edges_from ON memory_edges(from_node_id);
CREATE INDEX idx_memory_edges_to ON memory_edges(to_node_id);

CREATE VIRTUAL TABLE memory_nodes_fts USING fts5(
    label,
    content,
    content = 'memory_nodes',
    content_rowid = 'id',
    tokenize = 'porter unicode61'
);
CREATE TRIGGER memory_nodes_fts_ai AFTER INSERT ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(rowid, label, content)
    VALUES (new.id, new.label, new.content);
END;
CREATE TRIGGER memory_nodes_fts_ad AFTER DELETE ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(memory_nodes_fts, rowid, label, content)
    VALUES ('delete', old.id, old.label, old.content);
END;
CREATE TRIGGER memory_nodes_fts_au AFTER UPDATE OF label, content ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(memory_nodes_fts, rowid, label, content)
    VALUES ('delete', old.id, old.label, old.content);
    INSERT INTO memory_nodes_fts(rowid, label, content)
    VALUES (new.id, new.label, new.content);
END;

-- [M#] references used by an answer, with an immutable snapshot so a later
-- graph rebuild, rename or source deletion never changes what an old [M1] meant.
CREATE TABLE message_memory_refs (
    id                 INTEGER PRIMARY KEY,
    message_id         INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    label              INTEGER NOT NULL CHECK (label >= 1),
    ordinal            INTEGER NOT NULL CHECK (ordinal >= 0),
    node_key           TEXT NOT NULL,      -- identity of the remembered node
    kind               TEXT NOT NULL,      -- 'exchange' | 'correction' | 'question'
    provenance         TEXT NOT NULL,
    conversation_id    INTEGER,            -- identity only
    conversation_title TEXT NOT NULL,
    question           TEXT NOT NULL,
    excerpt            TEXT NOT NULL,
    outcome            TEXT,
    correction         TEXT,
    sources            TEXT NOT NULL DEFAULT '[]',  -- JSON: cited filename/page snapshots
    remembered_at      TEXT NOT NULL,      -- when the remembered item was created
    UNIQUE (message_id, label)
);
CREATE INDEX idx_memory_refs_message ON message_memory_refs(message_id);
