-- Phase 7: notes, note links, chat full-text search, notes in graph memory.
--
-- Notes are user-authored and authoritative for themselves; links carry an
-- immutable snapshot of what was linked (no FK to the linked item), so a note
-- stays readable after a source, message or memory node disappears.

CREATE TABLE notes (
    id          INTEGER PRIMARY KEY,
    notebook_id INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title       TEXT NOT NULL CHECK (trim(title) <> '' AND length(title) <= 200),
    content     TEXT NOT NULL DEFAULT '' CHECK (length(CAST(content AS BLOB)) <= 1048576),
    revision    INTEGER NOT NULL DEFAULT 1,   -- bumped on every save (conflict check)
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_notes_notebook ON notes(notebook_id, updated_at);

CREATE VIRTUAL TABLE notes_fts USING fts5(
    title, content, content = 'notes', content_rowid = 'id',
    tokenize = 'porter unicode61'
);
CREATE TRIGGER notes_fts_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;
CREATE TRIGGER notes_fts_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
END;
CREATE TRIGGER notes_fts_au AFTER UPDATE OF title, content ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content)
    VALUES ('delete', old.id, old.title, old.content);
    INSERT INTO notes_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;

CREATE TABLE note_links (
    id              INTEGER PRIMARY KEY,
    note_id         INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    link_type       TEXT NOT NULL CHECK (link_type IN ('source_chunk', 'message', 'memory_node')),
    target_key      TEXT NOT NULL,    -- 'chunk:12' | 'message:5' | 'memory:concept:matrix'
    source_id       INTEGER,          -- identity values only, never FKs
    chunk_id        INTEGER,
    message_id      INTEGER,
    conversation_id INTEGER,
    memory_key      TEXT,
    label           TEXT NOT NULL,    -- snapshot: filename / conversation title / node label
    excerpt         TEXT NOT NULL DEFAULT '',
    page_no         INTEGER,
    chunk_ord       INTEGER,
    details         TEXT NOT NULL DEFAULT '{}',  -- JSON snapshot (e.g. a message's citations)
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (note_id, target_key)
);
CREATE INDEX idx_note_links_note ON note_links(note_id);

-- Chat search: message text and conversation titles (never prompts).
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content, content = 'messages', content_rowid = 'id', tokenize = 'porter unicode61'
);
CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER messages_fts_au AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
INSERT INTO messages_fts(messages_fts) VALUES ('rebuild');

CREATE VIRTUAL TABLE conversations_fts USING fts5(
    title, content = 'conversations', content_rowid = 'id', tokenize = 'porter unicode61'
);
CREATE TRIGGER conversations_fts_ai AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, title) VALUES (new.id, new.title);
END;
CREATE TRIGGER conversations_fts_ad AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, title)
    VALUES ('delete', old.id, old.title);
END;
CREATE TRIGGER conversations_fts_au AFTER UPDATE OF title ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, title)
    VALUES ('delete', old.id, old.title);
    INSERT INTO conversations_fts(rowid, title) VALUES (new.id, new.title);
END;
INSERT INTO conversations_fts(conversations_fts) VALUES ('rebuild');

-- Graph memory gains notes: node type 'note', provenance 'user_note',
-- relations note_mentions / note_supported_by. CHECK constraints cannot be
-- altered, so both tables are rebuilt with every row and id preserved (the
-- migration runner disables foreign keys while migrating).
CREATE TABLE memory_nodes_new (
    id              INTEGER PRIMARY KEY,
    notebook_id     INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    key             TEXT NOT NULL,
    node_type       TEXT NOT NULL CHECK (node_type IN (
                        'conversation', 'question', 'answer', 'concept',
                        'source_evidence', 'correction', 'note')),
    label           TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    provenance      TEXT NOT NULL CHECK (provenance IN (
                        'user_statement', 'assistant_response', 'source_backed',
                        'user_correction', 'user_note', 'derived')),
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    message_id      INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    note_id         INTEGER REFERENCES notes(id) ON DELETE CASCADE,
    source_id       INTEGER,
    chunk_id        INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (notebook_id, key)
);
INSERT INTO memory_nodes_new (id, notebook_id, key, node_type, label, content, provenance,
                              conversation_id, message_id, source_id, chunk_id,
                              created_at, updated_at)
SELECT id, notebook_id, key, node_type, label, content, provenance, conversation_id,
       message_id, source_id, chunk_id, created_at, updated_at
FROM memory_nodes;

CREATE TABLE memory_edges_new (
    id           INTEGER PRIMARY KEY,
    notebook_id  INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    from_node_id INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    to_node_id   INTEGER NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    relation     TEXT NOT NULL CHECK (relation IN (
                     'contains', 'answered_by', 'followed_by', 'mentions',
                     'supported_by', 'corrects', 'note_mentions', 'note_supported_by')),
    provenance   TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (from_node_id <> to_node_id),
    UNIQUE (from_node_id, to_node_id, relation)
);
INSERT INTO memory_edges_new (id, notebook_id, from_node_id, to_node_id, relation,
                              provenance, created_at)
SELECT id, notebook_id, from_node_id, to_node_id, relation, provenance, created_at
FROM memory_edges;

DROP TABLE memory_edges;
DROP TABLE memory_nodes;
ALTER TABLE memory_nodes_new RENAME TO memory_nodes;
ALTER TABLE memory_edges_new RENAME TO memory_edges;
CREATE INDEX idx_memory_nodes_type ON memory_nodes(notebook_id, node_type);
CREATE INDEX idx_memory_nodes_message ON memory_nodes(message_id);
CREATE INDEX idx_memory_nodes_conversation ON memory_nodes(conversation_id);
CREATE INDEX idx_memory_nodes_note ON memory_nodes(note_id);
CREATE INDEX idx_memory_edges_from ON memory_edges(from_node_id);
CREATE INDEX idx_memory_edges_to ON memory_edges(to_node_id);

-- memory_nodes_fts keeps its rows (ids preserved); re-attach its triggers.
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
INSERT INTO memory_nodes_fts(memory_nodes_fts) VALUES ('rebuild');
