-- Phase 2: user-editable application settings (key/value).
-- Precedence at runtime: LITENOTEBOOK_* environment variable > this table > default.

CREATE TABLE settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
