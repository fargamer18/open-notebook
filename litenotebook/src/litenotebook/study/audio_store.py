"""Audio Overviews in SQLite (migration 0010) and their files on disk.

Reading an overview, its transcript, chapters or audio uses only this module
and the file system: no retrieval, no MiniLM, no LLM, no speech engine.
Every lookup is scoped by notebook id.

Audio files: ``notebooks/<notebook>/audio/<overview>/<asset>.wav`` — built
from integer ids only.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litenotebook.config import Settings
from litenotebook.platform_utils.filesystem import is_within, remove_tree
from litenotebook.study.audio_model import (
    FORMAT_LABELS,
    INTERRUPTED,
    SPEAKER_LABELS,
    SPEECH_INTERRUPTED,
    Script,
    estimate_seconds,
)
from litenotebook.study.model import KIND_PREFIX

NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
SCRIPT_STAGES = ("pending", "retrieving", "generating_script")


class AudioError(ValueError):
    pass


class OverviewNotFoundError(AudioError):
    pass


class AssetNotFoundError(AudioError):
    pass


@dataclass
class EvidenceRef:
    label: str
    kind: str
    title: str
    excerpt: str
    source_id: int | None
    chunk_id: int | None
    page_no: int | None
    note_id: int | None
    message_id: int | None
    conversation_id: int | None
    exists: bool = False
    chunk_exists: bool = False


@dataclass
class Turn:
    id: int
    position: int
    speaker: str
    text: str
    refs: list[str]

    @property
    def speaker_label(self) -> str:
        return SPEAKER_LABELS.get(self.speaker, self.speaker)


@dataclass
class Section:
    id: int
    position: int
    title: str
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Asset:
    id: int
    overview_id: int
    status: str
    engine: str
    voices: dict[str, str]
    sample_rate: int | None
    duration_s: float | None
    size_bytes: int | None
    sha256: str | None
    chapters: list[dict[str, Any]]
    turn_times: list[list[float]]
    error: str | None
    meta: dict[str, Any]
    created_at: str
    completed_at: str | None


@dataclass
class Overview:
    id: int
    notebook_id: int
    title: str
    format: str
    target_minutes: int
    focus: str
    material: dict[str, Any]
    status: str
    stage: str
    message: str | None
    error: str | None
    model: str | None
    word_count: int
    words_per_minute: int
    meta: dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: str | None
    latest_asset: Asset | None = None

    @property
    def format_label(self) -> str:
        return FORMAT_LABELS.get(self.format, self.format)

    @property
    def estimated_seconds(self) -> int:
        return estimate_seconds(self.word_count, self.words_per_minute)

    @property
    def source_names(self) -> list[str]:
        return [s.get("filename", "") for s in self.material.get("sources", [])]

    @property
    def has_script(self) -> bool:
        return self.status in ("script_ready", "synthesizing", "ready")

    @property
    def busy(self) -> bool:
        return self.status in (*SCRIPT_STAGES, "synthesizing")


def _overview(row: sqlite3.Row) -> Overview:
    return Overview(
        id=row["id"],
        notebook_id=row["notebook_id"],
        title=row["title"],
        format=row["format"],
        target_minutes=row["target_minutes"],
        focus=row["focus"],
        material=json.loads(row["material"] or "{}"),
        status=row["status"],
        stage=row["stage"],
        message=row["message"],
        error=row["error"],
        model=row["model"],
        word_count=row["word_count"],
        words_per_minute=row["words_per_minute"],
        meta=json.loads(row["meta"] or "{}"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _asset(row: sqlite3.Row) -> Asset:
    return Asset(
        id=row["id"],
        overview_id=row["overview_id"],
        status=row["status"],
        engine=row["engine"],
        voices=json.loads(row["voices"] or "{}"),
        sample_rate=row["sample_rate"],
        duration_s=row["duration_s"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        chapters=json.loads(row["chapters"] or "[]"),
        turn_times=json.loads(row["turn_times"] or "[]"),
        error=row["error"],
        meta=json.loads(row["meta"] or "{}"),
        created_at=row["created_at"],
        completed_at=row["completed_at"],
    )


def _latest_assets(conn: sqlite3.Connection, ids: list[int]) -> dict[int, Asset]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    out: dict[int, Asset] = {}
    for row in conn.execute(
        f"SELECT * FROM audio_assets WHERE overview_id IN ({marks}) "
        "ORDER BY (status = 'ready') DESC, id DESC",
        ids,
    ):
        out.setdefault(row["overview_id"], _asset(row))
    return out


def list_overviews(
    conn: sqlite3.Connection, notebook_id: int, limit: int = 200
) -> list[Overview]:
    rows = conn.execute(
        "SELECT * FROM audio_overviews WHERE notebook_id = ? "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (notebook_id, limit),
    ).fetchall()
    items = [_overview(r) for r in rows]
    assets = _latest_assets(conn, [o.id for o in items])
    for o in items:
        o.latest_asset = assets.get(o.id)
    return items


def get_overview(
    conn: sqlite3.Connection, notebook_id: int, overview_id: int
) -> Overview:
    row = conn.execute(
        "SELECT * FROM audio_overviews WHERE id = ? AND notebook_id = ?",
        (overview_id, notebook_id),
    ).fetchone()
    if row is None:
        raise OverviewNotFoundError(f"Audio overview {overview_id} not found")
    o = _overview(row)
    o.latest_asset = _latest_assets(conn, [o.id]).get(o.id)
    return o


def sections(conn: sqlite3.Connection, overview_id: int) -> list[Section]:
    out: dict[int, Section] = {}
    for r in conn.execute(
        "SELECT id, position, title FROM audio_sections WHERE overview_id = ? "
        "ORDER BY position",
        (overview_id,),
    ):
        out[r["id"]] = Section(r["id"], r["position"], r["title"])
    if out:
        marks = ",".join("?" * len(out))
        for r in conn.execute(
            f"SELECT * FROM audio_turns WHERE section_id IN ({marks}) "
            "ORDER BY section_id, position",
            list(out),
        ):
            out[r["section_id"]].turns.append(
                Turn(
                    r["id"],
                    r["position"],
                    r["speaker"],
                    r["text"],
                    json.loads(r["refs"] or "[]"),
                )
            )
    return sorted(out.values(), key=lambda s: s.position)


def evidence(conn: sqlite3.Connection, overview_id: int) -> dict[str, EvidenceRef]:
    out: dict[str, EvidenceRef] = {}
    for r in conn.execute(
        """SELECT e.*,
             CASE e.kind
               WHEN 'source' THEN EXISTS (SELECT 1 FROM sources s
                    WHERE s.id = e.source_id AND s.status = 'ready'
                    AND s.notebook_id = o.notebook_id)
               WHEN 'note' THEN EXISTS (SELECT 1 FROM notes n
                    WHERE n.id = e.note_id AND n.notebook_id = o.notebook_id)
               ELSE EXISTS (SELECT 1 FROM conversations c
                    WHERE c.id = e.conversation_id AND c.notebook_id = o.notebook_id)
             END AS live,
             EXISTS (SELECT 1 FROM chunks c WHERE c.id = e.chunk_id
                    AND c.source_id = e.source_id) AS chunk_live
           FROM audio_evidence e JOIN audio_overviews o ON o.id = e.overview_id
           WHERE e.overview_id = ? ORDER BY e.id""",
        (overview_id,),
    ):
        out[r["label"]] = EvidenceRef(
            label=r["label"],
            kind=r["kind"],
            title=r["title"],
            excerpt=r["excerpt"],
            source_id=r["source_id"],
            chunk_id=r["chunk_id"],
            page_no=r["page_no"],
            note_id=r["note_id"],
            message_id=r["message_id"],
            conversation_id=r["conversation_id"],
            exists=bool(r["live"]),
            chunk_exists=bool(r["chunk_live"]),
        )
    return out


def list_assets(conn: sqlite3.Connection, overview_id: int) -> list[Asset]:
    return [
        _asset(r)
        for r in conn.execute(
            "SELECT * FROM audio_assets WHERE overview_id = ? ORDER BY id DESC",
            (overview_id,),
        )
    ]


def get_asset(
    conn: sqlite3.Connection, notebook_id: int, overview_id: int, asset_id: int
) -> Asset:
    get_overview(conn, notebook_id, overview_id)
    row = conn.execute(
        "SELECT * FROM audio_assets WHERE id = ? AND overview_id = ?",
        (asset_id, overview_id),
    ).fetchone()
    if row is None:
        raise AssetNotFoundError("Audio not found")
    return _asset(row)


# --- files ------------------------------------------------------------------------


def overview_dir(settings: Settings, notebook_id: int, overview_id: int) -> Path:
    return settings.notebook_dir(int(notebook_id)) / "audio" / str(int(overview_id))


def asset_path(
    settings: Settings, notebook_id: int, overview_id: int, asset_id: int
) -> Path:
    folder = overview_dir(settings, notebook_id, overview_id)
    path = folder / f"{int(asset_id)}.wav"
    if not is_within(path, settings.notebook_dir(int(notebook_id))):
        raise AssetNotFoundError("Audio not found")
    return path


# --- writing ----------------------------------------------------------------------


def create_overview(
    conn: sqlite3.Connection,
    notebook_id: int,
    *,
    title: str,
    fmt: str,
    minutes: int,
    focus: str,
    material: dict[str, Any],
    wpm: int,
) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO audio_overviews (notebook_id, title, format, target_minutes, "
            "focus, material, words_per_minute) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                notebook_id,
                title,
                fmt,
                minutes,
                focus,
                json.dumps(material, ensure_ascii=False),
                wpm,
            ),
        )
    assert cur.lastrowid is not None
    return cur.lastrowid


def set_stage(
    conn: sqlite3.Connection, overview_id: int, status: str, stage: str | None = None
) -> None:
    with conn:
        conn.execute(
            f"UPDATE audio_overviews SET status = ?, stage = ?, updated_at = {NOW_SQL} "
            "WHERE id = ? AND status IN "
            "('pending', 'retrieving', 'generating_script')",
            (status, stage or status, overview_id),
        )


def fail(
    conn: sqlite3.Connection,
    overview_id: int,
    error: str,
    *,
    status: str = "failed",
    meta: dict[str, Any] | None = None,
) -> None:
    """End a script generation without a transcript."""
    with conn:
        conn.execute("DELETE FROM audio_sections WHERE overview_id = ?", (overview_id,))
        conn.execute("DELETE FROM audio_evidence WHERE overview_id = ?", (overview_id,))
        conn.execute(
            f"UPDATE audio_overviews SET status = ?, stage = ?, error = ?, meta = ?, "
            f"word_count = 0, updated_at = {NOW_SQL}, completed_at = {NOW_SQL} "
            "WHERE id = ?",
            (status, status, error, json.dumps(meta or {}), overview_id),
        )


def save_script(
    conn: sqlite3.Connection,
    overview_id: int,
    script: Script,
    *,
    message: str | None,
    model: str | None,
    meta: dict[str, Any],
) -> None:
    """Persist a validated script in one transaction; labels are renumbered
    per kind in order of first use (S1, S2, N1 …)."""
    order: list[str] = []
    for s in script.sections:
        for label in s.refs:
            if label not in order:
                order.append(label)
    counters: dict[str, int] = {}
    rename: dict[str, str] = {}
    for label in order:
        item = script.evidence[label]
        counters[item.kind] = counters.get(item.kind, 0) + 1
        rename[label] = f"{KIND_PREFIX[item.kind]}{counters[item.kind]}"
    with conn:
        row = conn.execute(
            "SELECT status FROM audio_overviews WHERE id = ?", (overview_id,)
        ).fetchone()
        if row is None or row[0] not in ("retrieving", "generating_script", "pending"):
            raise AudioError("This overview is no longer being generated")
        for old, new in rename.items():
            e = script.evidence[old]
            conn.execute(
                "INSERT INTO audio_evidence (overview_id, label, kind, source_id, "
                "chunk_id, page_no, chunk_ord, note_id, message_id, conversation_id, "
                "title, excerpt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    overview_id,
                    new,
                    e.kind,
                    e.source_id,
                    e.chunk_id,
                    e.page_no,
                    e.chunk_ord,
                    e.note_id,
                    e.message_id,
                    e.conversation_id,
                    e.title,
                    e.text,
                ),
            )
        for position, s in enumerate(script.sections, 1):
            cur = conn.execute(
                "INSERT INTO audio_sections (overview_id, position, title) "
                "VALUES (?, ?, ?)",
                (overview_id, position, s.title),
            )
            conn.executemany(
                "INSERT INTO audio_turns (section_id, position, speaker, text, refs) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        cur.lastrowid,
                        i,
                        t.speaker,
                        t.text,
                        json.dumps([rename[r] for r in t.refs]),
                    )
                    for i, t in enumerate(s.turns, 1)
                ],
            )
        conn.execute(
            "UPDATE audio_overviews SET status = 'script_ready', "
            "stage = 'script_ready', title = ?, word_count = ?, message = ?, "
            "model = ?, meta = ?, "
            f"updated_at = {NOW_SQL}, completed_at = {NOW_SQL} WHERE id = ?",
            (script.title, script.words, message, model, json.dumps(meta), overview_id),
        )


def start_asset(
    conn: sqlite3.Connection, overview_id: int, engine: str, voices: dict[str, str]
) -> int:
    with conn:
        updated = conn.execute(
            "UPDATE audio_overviews SET status = 'synthesizing', "
            f"stage = 'synthesizing', error = NULL, updated_at = {NOW_SQL} "
            "WHERE id = ? "
            "AND status IN ('script_ready', 'ready')",
            (overview_id,),
        ).rowcount
        if updated != 1:
            raise AudioError("This overview has no finished transcript to speak")
        cur = conn.execute(
            "INSERT INTO audio_assets (overview_id, engine, voices) VALUES (?, ?, ?)",
            (overview_id, engine, json.dumps(voices)),
        )
    assert cur.lastrowid is not None
    return cur.lastrowid


def set_synth_stage(conn: sqlite3.Connection, overview_id: int, stage: str) -> None:
    with conn:
        conn.execute(
            f"UPDATE audio_overviews SET stage = ?, updated_at = {NOW_SQL} "
            "WHERE id = ? AND status = 'synthesizing'",
            (stage, overview_id),
        )


def finish_asset(
    conn: sqlite3.Connection,
    overview_id: int,
    asset_id: int,
    *,
    sample_rate: int,
    duration_s: float,
    size_bytes: int,
    sha256: str,
    chapters: list[dict[str, Any]],
    turn_times: list[list[float]],
    meta: dict[str, Any],
) -> None:
    with conn:
        conn.execute(
            "UPDATE audio_assets SET status = 'ready', sample_rate = ?, "
            "duration_s = ?, size_bytes = ?, sha256 = ?, chapters = ?, "
            "turn_times = ?, meta = ?, "
            f"completed_at = {NOW_SQL} WHERE id = ?",
            (
                sample_rate,
                duration_s,
                size_bytes,
                sha256,
                json.dumps(chapters),
                json.dumps(turn_times),
                json.dumps(meta),
                asset_id,
            ),
        )
        conn.execute(
            f"UPDATE audio_overviews SET status = 'ready', stage = 'ready', "
            f"updated_at = {NOW_SQL} WHERE id = ?",
            (overview_id,),
        )


def end_asset(
    conn: sqlite3.Connection,
    overview_id: int,
    asset_id: int,
    *,
    status: str,
    error: str,
    meta: dict[str, Any] | None = None,
) -> None:
    """Speech failed or was cancelled: the transcript stays."""
    with conn:
        conn.execute(
            f"UPDATE audio_assets SET status = ?, error = ?, meta = ?, "
            f"completed_at = {NOW_SQL} WHERE id = ?",
            (status, error, json.dumps(meta or {}), asset_id),
        )
        has_ready = conn.execute(
            "SELECT 1 FROM audio_assets WHERE overview_id = ? AND status = 'ready'",
            (overview_id,),
        ).fetchone()
        conn.execute(
            f"UPDATE audio_overviews SET status = ?, stage = ?, updated_at = {NOW_SQL} "
            "WHERE id = ?",
            (
                "ready" if has_ready else "script_ready",
                "ready" if has_ready else "script_ready",
                overview_id,
            ),
        )


def recover(conn: sqlite3.Connection, settings: Settings) -> tuple[int, int]:
    """Startup: unfinished script generations fail; an interrupted synthesis
    keeps its transcript (the half-written audio file is removed)."""
    scripts = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM audio_overviews WHERE status IN "
            "('pending', 'retrieving', 'generating_script')"
        )
    ]
    for oid in scripts:
        fail(conn, oid, INTERRUPTED)
    speech = conn.execute(
        "SELECT a.id, a.overview_id, o.notebook_id FROM audio_assets a "
        "JOIN audio_overviews o ON o.id = a.overview_id "
        "WHERE a.status = 'synthesizing'"
    ).fetchall()
    for aid, oid, nb in speech:
        with contextlib.suppress(OSError, AssetNotFoundError):
            asset_path(settings, nb, oid, aid).unlink(missing_ok=True)
        end_asset(conn, oid, aid, status="failed", error=SPEECH_INTERRUPTED)
    with conn:
        conn.execute(
            "UPDATE audio_overviews SET status = 'script_ready', "
            f"stage = 'script_ready', updated_at = {NOW_SQL} "
            "WHERE status = 'synthesizing'"
        )
    return len(scripts), len(speech)


def delete_overview(
    conn: sqlite3.Connection, settings: Settings, notebook_id: int, overview_id: int
) -> Overview:
    """Delete an overview with its transcript, evidence snapshots and every
    audio file. Sources, notes and chats are not affected."""
    o = get_overview(conn, notebook_id, overview_id)
    if o.busy:
        raise AudioError("This overview is still being generated; cancel it first")
    with conn:
        conn.execute("DELETE FROM audio_overviews WHERE id = ?", (overview_id,))
    folder = overview_dir(settings, notebook_id, overview_id)
    if folder.exists() and is_within(folder, settings.notebook_dir(notebook_id)):
        remove_tree(folder)
    return o
