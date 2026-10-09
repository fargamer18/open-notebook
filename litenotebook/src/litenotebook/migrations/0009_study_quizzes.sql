-- Phase 13: Study → source-grounded quizzes.
--
-- A quiz is immutable once generated: its questions, options and evidence
-- snapshots can be inserted but never updated (triggers below), so every
-- attempt keeps pointing at exactly what was asked. "Generate again" makes a
-- new quiz. Evidence rows are snapshots: source/chunk/note/message ids are
-- identity values only (no foreign keys), so a quiz stays reviewable after a
-- source, note or conversation is deleted.
--
-- Attempts: one row per run through a quiz ('full') or through the missed
-- questions of an earlier attempt ('mistakes'). Each attempt has one item
-- per question, created up front in display order; an item is answered once
-- (answered_at is set, then the selection can no longer change).
--
-- study_concept_stats is derived from answered items (it can be rebuilt at
-- any time) and holds only counts, the last result and the current streak.

CREATE TABLE quizzes (
    id              INTEGER PRIMARY KEY,
    notebook_id     INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title           TEXT NOT NULL CHECK (trim(title) <> '' AND length(title) <= 200),
    status          TEXT NOT NULL DEFAULT 'generating'
                    CHECK (status IN ('generating', 'ready', 'failed', 'cancelled')),
    stage           TEXT NOT NULL DEFAULT 'queued',
    difficulty      TEXT NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard', 'exam')),
    question_types  TEXT NOT NULL,                 -- JSON list, e.g. ["single","true_false"]
    requested_count INTEGER NOT NULL CHECK (requested_count IN (5, 10, 15, 20)),
    question_count  INTEGER NOT NULL DEFAULT 0 CHECK (question_count >= 0),
    focus           TEXT NOT NULL DEFAULT '' CHECK (length(focus) <= 500),
    material        TEXT NOT NULL DEFAULT '{}',    -- JSON snapshot of what was selected
    message         TEXT,                          -- user-facing outcome note
    error           TEXT,                          -- user-facing; no paths or keys
    model           TEXT,
    meta            TEXT,                          -- JSON: timings and counts
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at    TEXT
);
CREATE INDEX idx_quizzes_notebook ON quizzes(notebook_id, created_at);

CREATE TABLE quiz_questions (
    id          INTEGER PRIMARY KEY,
    quiz_id     INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL CHECK (position >= 1),
    qtype       TEXT NOT NULL CHECK (qtype IN ('single', 'multiple', 'true_false')),
    question    TEXT NOT NULL CHECK (length(question) BETWEEN 1 AND 600),
    explanation TEXT NOT NULL CHECK (length(explanation) BETWEEN 1 AND 1200),
    hint        TEXT NOT NULL DEFAULT '' CHECK (length(hint) <= 400),
    difficulty  TEXT NOT NULL,
    concepts    TEXT NOT NULL DEFAULT '[]',        -- JSON [{"key": …, "label": …}]
    UNIQUE (quiz_id, position)
);

CREATE TABLE quiz_question_options (
    id          INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL CHECK (position BETWEEN 1 AND 4),
    text        TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND 300),
    is_correct  INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
    UNIQUE (question_id, position)
);

CREATE TABLE quiz_evidence (
    id              INTEGER PRIMARY KEY,
    question_id     INTEGER NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL CHECK (ordinal >= 1),
    kind            TEXT NOT NULL CHECK (kind IN ('source', 'note', 'chat')),
    source_id       INTEGER,   -- identity only (the source may be deleted later)
    chunk_id        INTEGER,   -- identity only
    page_no         INTEGER,
    chunk_ord       INTEGER,
    note_id         INTEGER,   -- identity only
    message_id      INTEGER,   -- identity only
    conversation_id INTEGER,   -- identity only
    label           TEXT NOT NULL,   -- file name / note title / conversation title
    excerpt         TEXT NOT NULL CHECK (length(excerpt) BETWEEN 1 AND 4000),
    UNIQUE (question_id, ordinal)
);
CREATE INDEX idx_quiz_evidence_question ON quiz_evidence(question_id);

CREATE TABLE quiz_attempts (
    id                INTEGER PRIMARY KEY,
    quiz_id           INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
    mode              TEXT NOT NULL CHECK (mode IN ('full', 'mistakes')),
    parent_attempt_id INTEGER REFERENCES quiz_attempts(id) ON DELETE SET NULL,
    status            TEXT NOT NULL DEFAULT 'in_progress'
                      CHECK (status IN ('in_progress', 'completed', 'abandoned')),
    total             INTEGER NOT NULL CHECK (total >= 1),
    score             INTEGER CHECK (score IS NULL OR score BETWEEN 0 AND total),
    started_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at      TEXT
);
CREATE INDEX idx_quiz_attempts_quiz ON quiz_attempts(quiz_id, started_at);

CREATE TABLE quiz_attempt_items (
    id          INTEGER PRIMARY KEY,
    attempt_id  INTEGER NOT NULL REFERENCES quiz_attempts(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL CHECK (position >= 1),
    selected    TEXT,                              -- JSON option positions; NULL = unanswered
    is_correct  INTEGER CHECK (is_correct IS NULL OR is_correct IN (0, 1)),
    hint_used   INTEGER NOT NULL DEFAULT 0 CHECK (hint_used IN (0, 1)),
    answered_at TEXT,
    UNIQUE (attempt_id, position),
    UNIQUE (attempt_id, question_id)
);
CREATE INDEX idx_quiz_items_question ON quiz_attempt_items(question_id);

CREATE TABLE study_concept_stats (
    id              INTEGER PRIMARY KEY,
    notebook_id     INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    concept_key     TEXT NOT NULL CHECK (length(concept_key) BETWEEN 1 AND 80),
    label           TEXT NOT NULL,
    times_seen      INTEGER NOT NULL DEFAULT 0 CHECK (times_seen >= 0),
    correct_count   INTEGER NOT NULL DEFAULT 0 CHECK (correct_count >= 0),
    incorrect_count INTEGER NOT NULL DEFAULT 0 CHECK (incorrect_count >= 0),
    streak          INTEGER NOT NULL DEFAULT 0,  -- +n correct / -n incorrect in a row
    last_result     TEXT CHECK (last_result IS NULL OR last_result IN ('correct', 'incorrect')),
    last_seen       TEXT,
    UNIQUE (notebook_id, concept_key)
);

-- Generated content never changes after it is saved.
CREATE TRIGGER quiz_questions_immutable BEFORE UPDATE ON quiz_questions BEGIN
    SELECT RAISE(ABORT, 'quiz questions are immutable');
END;
CREATE TRIGGER quiz_options_immutable BEFORE UPDATE ON quiz_question_options BEGIN
    SELECT RAISE(ABORT, 'quiz options are immutable');
END;
CREATE TRIGGER quiz_evidence_immutable BEFORE UPDATE ON quiz_evidence BEGIN
    SELECT RAISE(ABORT, 'quiz evidence is immutable');
END;
-- An answer is final: once answered_at is set the item cannot be changed.
CREATE TRIGGER quiz_items_answer_once BEFORE UPDATE OF selected, is_correct, answered_at
ON quiz_attempt_items WHEN old.answered_at IS NOT NULL BEGIN
    SELECT RAISE(ABORT, 'this question was already answered');
END;
