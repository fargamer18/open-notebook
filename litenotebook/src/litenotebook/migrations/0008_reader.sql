-- Phase 8: PDF reader — reading position, highlights and annotations.
--
-- Policy: reader annotations belong to their source and are deleted with it
-- (ON DELETE CASCADE). Anything copied into a note keeps its own snapshot in
-- note_links, so notes stay readable after the source is gone. The original
-- PDF is never modified; rendered pages live in a derived, deletable cache.
-- Rectangles use page coordinates normalised to 0..1 of the rendered
-- (rotation-applied) page, so they fit any render size.

CREATE TABLE reading_state (
    source_id    INTEGER PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
    notebook_id  INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    current_page INTEGER NOT NULL DEFAULT 1 CHECK (current_page >= 1),
    zoom         REAL NOT NULL DEFAULT 1.0 CHECK (zoom BETWEEN 0.25 AND 4.0),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE annotations (
    id              INTEGER PRIMARY KEY,
    notebook_id     INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    source_id       INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    page_no         INTEGER NOT NULL CHECK (page_no >= 1),
    annotation_type TEXT NOT NULL CHECK (annotation_type IN ('highlight', 'comment')),
    selected_text   TEXT NOT NULL DEFAULT '' CHECK (length(selected_text) <= 5000),
    comment         TEXT NOT NULL DEFAULT '' CHECK (length(comment) <= 10000),
    color           TEXT NOT NULL DEFAULT 'yellow'
                    CHECK (color IN ('yellow', 'green', 'blue', 'pink')),
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_annotations_source_page ON annotations(source_id, page_no);
CREATE INDEX idx_annotations_notebook ON annotations(notebook_id, updated_at);

CREATE TABLE annotation_rects (
    id            INTEGER PRIMARY KEY,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
    page_no       INTEGER NOT NULL CHECK (page_no >= 1),
    x0 REAL NOT NULL, y0 REAL NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL,
    ordinal       INTEGER NOT NULL CHECK (ordinal >= 0),
    CHECK (x0 >= 0 AND y0 >= 0 AND x1 <= 1 AND y1 <= 1 AND x0 < x1 AND y0 < y1)
);
CREATE INDEX idx_annotation_rects_annotation ON annotation_rects(annotation_id);

CREATE VIRTUAL TABLE annotations_fts USING fts5(
    selected_text, comment, content = 'annotations', content_rowid = 'id',
    tokenize = 'porter unicode61'
);
CREATE TRIGGER annotations_fts_ai AFTER INSERT ON annotations BEGIN
    INSERT INTO annotations_fts(rowid, selected_text, comment)
    VALUES (new.id, new.selected_text, new.comment);
END;
CREATE TRIGGER annotations_fts_ad AFTER DELETE ON annotations BEGIN
    INSERT INTO annotations_fts(annotations_fts, rowid, selected_text, comment)
    VALUES ('delete', old.id, old.selected_text, old.comment);
END;
CREATE TRIGGER annotations_fts_au AFTER UPDATE OF selected_text, comment ON annotations BEGIN
    INSERT INTO annotations_fts(annotations_fts, rowid, selected_text, comment)
    VALUES ('delete', old.id, old.selected_text, old.comment);
    INSERT INTO annotations_fts(rowid, selected_text, comment)
    VALUES (new.id, new.selected_text, new.comment);
END;

-- note_links gains link_type 'annotation' (+ an annotation_id identity value).
-- CHECK constraints cannot be altered: rebuild with every row and id kept.
CREATE TABLE note_links_new (
    id              INTEGER PRIMARY KEY,
    note_id         INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    link_type       TEXT NOT NULL CHECK (link_type IN (
                        'source_chunk', 'message', 'memory_node', 'annotation')),
    target_key      TEXT NOT NULL,
    source_id       INTEGER,
    chunk_id        INTEGER,
    message_id      INTEGER,
    conversation_id INTEGER,
    memory_key      TEXT,
    annotation_id   INTEGER,
    label           TEXT NOT NULL,
    excerpt         TEXT NOT NULL DEFAULT '',
    page_no         INTEGER,
    chunk_ord       INTEGER,
    details         TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (note_id, target_key)
);
INSERT INTO note_links_new (id, note_id, link_type, target_key, source_id, chunk_id,
                            message_id, conversation_id, memory_key, label, excerpt,
                            page_no, chunk_ord, details, created_at)
SELECT id, note_id, link_type, target_key, source_id, chunk_id, message_id,
       conversation_id, memory_key, label, excerpt, page_no, chunk_ord, details,
       created_at
FROM note_links;
DROP TABLE note_links;
ALTER TABLE note_links_new RENAME TO note_links;
CREATE INDEX idx_note_links_note ON note_links(note_id);
