-- Phase 14: Study → Audio Overviews (grounded podcast scripts + local speech).
--
-- An overview's script is immutable once generated: sections, turns and
-- evidence snapshots can be inserted but never updated (triggers below).
-- "Regenerate" makes a new overview. Speaking the same script again with
-- other voices adds a new audio_assets row; the transcript never changes.
--
-- Evidence rows are snapshots; source/chunk/note/message ids are identity
-- values only (no foreign keys), so a transcript stays reviewable after a
-- source is deleted. Turn citations are labels ("S1", "N2") into this table.
--
-- Status (one column for the whole lifecycle):
--   pending → retrieving → generating_script → script_ready
--   script_ready → synthesizing → ready         (an audio asset exists)
--   synthesizing → script_ready                  (speech failed/cancelled:
--                                                 the transcript is kept)
--   any script stage → failed | cancelled        (no transcript was saved)
-- Audio files live at notebooks/<notebook id>/audio/<overview id>/<asset id>.wav
-- — paths are built from ids only, never from titles, voices or file names.

CREATE TABLE audio_overviews (
    id               INTEGER PRIMARY KEY,
    notebook_id      INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title            TEXT NOT NULL CHECK (trim(title) <> '' AND length(title) <= 200),
    format           TEXT NOT NULL
                     CHECK (format IN ('two_person', 'narrator', 'revision', 'deep_dive')),
    target_minutes   INTEGER NOT NULL CHECK (target_minutes IN (5, 10, 20)),
    focus            TEXT NOT NULL DEFAULT '' CHECK (length(focus) <= 500),
    material         TEXT NOT NULL DEFAULT '{}',   -- JSON snapshot of the selection
    status           TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                         'pending', 'retrieving', 'generating_script', 'script_ready',
                         'synthesizing', 'ready', 'failed', 'cancelled')),
    stage            TEXT NOT NULL DEFAULT 'pending',
    message          TEXT,                         -- user-facing outcome note
    error            TEXT,                         -- user-facing; no paths or keys
    model            TEXT,
    word_count       INTEGER NOT NULL DEFAULT 0 CHECK (word_count >= 0),
    words_per_minute INTEGER NOT NULL DEFAULT 150 CHECK (words_per_minute BETWEEN 80 AND 260),
    meta             TEXT,                         -- JSON: timings, calls, removed turns
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at     TEXT
);
CREATE INDEX idx_audio_overviews_notebook ON audio_overviews(notebook_id, created_at);

CREATE TABLE audio_sections (
    id          INTEGER PRIMARY KEY,
    overview_id INTEGER NOT NULL REFERENCES audio_overviews(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL CHECK (position >= 1),
    title       TEXT NOT NULL CHECK (length(title) BETWEEN 1 AND 200),
    UNIQUE (overview_id, position)
);

CREATE TABLE audio_turns (
    id         INTEGER PRIMARY KEY,
    section_id INTEGER NOT NULL REFERENCES audio_sections(id) ON DELETE CASCADE,
    position   INTEGER NOT NULL CHECK (position >= 1),
    speaker    TEXT NOT NULL CHECK (speaker IN ('A', 'B', 'narrator')),
    text       TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND 1500),
    refs       TEXT NOT NULL DEFAULT '[]',          -- JSON labels, e.g. ["S1","S3"]
    UNIQUE (section_id, position)
);

CREATE TABLE audio_evidence (
    id              INTEGER PRIMARY KEY,
    overview_id     INTEGER NOT NULL REFERENCES audio_overviews(id) ON DELETE CASCADE,
    label           TEXT NOT NULL,                  -- S1 / N1 / M1 within this overview
    kind            TEXT NOT NULL CHECK (kind IN ('source', 'note', 'chat')),
    source_id       INTEGER,   -- identity only (the source may be deleted later)
    chunk_id        INTEGER,   -- identity only
    page_no         INTEGER,
    chunk_ord       INTEGER,
    note_id         INTEGER,   -- identity only
    message_id      INTEGER,   -- identity only
    conversation_id INTEGER,   -- identity only
    title           TEXT NOT NULL,                  -- file name / note / conversation title
    excerpt         TEXT NOT NULL CHECK (length(excerpt) BETWEEN 1 AND 4000),
    UNIQUE (overview_id, label)
);

CREATE TABLE audio_assets (
    id           INTEGER PRIMARY KEY,
    overview_id  INTEGER NOT NULL REFERENCES audio_overviews(id) ON DELETE CASCADE,
    status       TEXT NOT NULL DEFAULT 'synthesizing'
                 CHECK (status IN ('synthesizing', 'ready', 'failed', 'cancelled')),
    engine       TEXT NOT NULL,                     -- e.g. "sherpa-onnx 1.13.8"
    voices       TEXT NOT NULL DEFAULT '{}',        -- JSON speaker → voice id
    sample_rate  INTEGER,
    duration_s   REAL,                              -- measured from the audio file
    size_bytes   INTEGER,
    sha256       TEXT,
    chapters     TEXT NOT NULL DEFAULT '[]',        -- JSON [{section, title, start_s}]
    turn_times   TEXT NOT NULL DEFAULT '[]',        -- JSON [[turn id, start_s, end_s]]
    error        TEXT,
    meta         TEXT,                              -- JSON: synthesis timings
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at TEXT
);
CREATE INDEX idx_audio_assets_overview ON audio_assets(overview_id, created_at);

CREATE TRIGGER audio_sections_immutable BEFORE UPDATE ON audio_sections BEGIN
    SELECT RAISE(ABORT, 'audio scripts are immutable');
END;
CREATE TRIGGER audio_turns_immutable BEFORE UPDATE ON audio_turns BEGIN
    SELECT RAISE(ABORT, 'audio scripts are immutable');
END;
CREATE TRIGGER audio_evidence_immutable BEFORE UPDATE ON audio_evidence BEGIN
    SELECT RAISE(ABORT, 'audio evidence is immutable');
END;
