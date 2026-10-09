"""Import notebooks from a validated archive, all-or-nothing.

Lifecycle for each notebook::

    validate the whole archive (reader.validate)        — nothing written yet
    stage PDFs into <data>/tmp/archives/import-<uuid>/  — re-hashed while copying
    BEGIN IMMEDIATE
      insert every row with new local ids (old → new maps per table)
      remap soft references and graph keys
      move staged PDFs to notebooks/<new>/sources/<new>.pdf
    COMMIT
    rebuild derived data: memory graph (deterministic), FAISS (from BLOBs)
    integrity report: FAISS n/n, memory graph, source files

Failure semantics: anything before COMMIT rolls the transaction back and
deletes the staged and already-moved files, so no notebook row or file is
left. A failure after COMMIT (derived rebuild or integrity check) deletes the
new notebook again through the normal ``delete_notebook`` path and reports
the stage that failed. Either way the import has no effect.

Soft references that point outside the archive (citations and note links to
sources deleted before the export) become values that can never match a
live row (0 for NOT NULL columns, NULL otherwise); their text snapshots stay.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from litenotebook.archive.format import TABLES, ArchiveError, Limits
from litenotebook.archive.integrity import NotebookReport, notebook_report
from litenotebook.archive.reader import ArchiveReader, NotebookInfo, VerifyReport
from litenotebook.config import EMBED_DIM, EMBED_MODEL, Settings
from litenotebook.db import schema_version
from litenotebook.notebooks import MAX_NAME_LENGTH
from litenotebook.platform_utils.filesystem import (
    create_exclusive,
    remove_tree,
    replace_file,
)

Progress = Callable[[str], None]
REEMBED_MESSAGE = (
    "Imported without usable embeddings ({why}); use Retry to re-embed this source"
)

# Graph-memory keys carry row ids: "<prefix>:<id>" → table that owns the id.
KEY_TABLES = {
    "conversation": "conversations",
    "question": "messages",
    "answer": "messages",
    "correction": "messages",
    "evidence": "message_citations",
    "note": "notes",
    "note_evidence": "note_links",
}
TARGET_TABLES = {"chunk": "chunks", "message": "messages", "annotation": "annotations"}


class ImportFailedError(ArchiveError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(
            f"Import failed while {stage}: {message}. Nothing was imported."
        )
        self.stage = stage


@dataclass
class ImportResult:
    notebook_id: int
    name: str
    original_name: str
    renamed: bool
    report: NotebookReport
    stages: list[tuple[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reembed_sources: int = 0
    seconds: float = 0.0


Maps = dict[str, dict[int, int]]


def remap_key(key: str, maps: Maps) -> str:
    """Rewrite a graph-memory key to the new ids; unknown ids become keys that
    can never match a live node."""
    prefix, sep, rest = key.partition(":")
    table = KEY_TABLES.get(prefix)
    if not sep or table is None or not rest.isdigit():
        return key  # e.g. "concept:…" has no id
    new = maps.get(table, {}).get(int(rest))
    return f"{prefix}:{new}" if new is not None else f"{prefix}:gone-{rest}"


def remap_target(link_type: str, key: str, maps: Maps) -> str:
    prefix, sep, rest = key.partition(":")
    if link_type == "memory_node" and prefix == "memory":
        return f"memory:{remap_key(rest, maps)}"
    table = TARGET_TABLES.get(prefix)
    if not sep or table is None or not rest.isdigit():
        return key
    new = maps.get(table, {}).get(int(rest))
    return f"{prefix}:{new}" if new is not None else f"{prefix}:gone-{rest}"


def _soft(maps: Maps, table: str, value: int | None, missing: int | None) -> int | None:
    if value is None:
        return None
    return maps.get(table, {}).get(value, missing)


def _remap_json_ids(raw: str | None, maps: Maps, list_key: str | None) -> str | None:
    """chunk_id / source_id inside snapshot JSON (citation lists)."""
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    items = data.get(list_key) if list_key and isinstance(data, dict) else data
    if not isinstance(items, list):
        return raw
    for item in items:
        if isinstance(item, dict):
            for col, table in (("chunk_id", "chunks"), ("source_id", "sources")):
                if isinstance(item.get(col), int):
                    item[col] = _soft(maps, table, item[col], None)
    return json.dumps(data, ensure_ascii=False)


def _check_json(raw: str, where: str) -> None:
    try:
        json.loads(raw)
    except ValueError as e:
        raise ArchiveError(f"Invalid data in {where}") from e


def unique_name(conn: sqlite3.Connection, base: str) -> str:
    """``base`` if free, else "base (Imported)", "base (Imported 2)", …"""
    base = " ".join(base.split())[:MAX_NAME_LENGTH] or "Imported notebook"

    def taken(name: str) -> bool:
        return (
            conn.execute("SELECT 1 FROM notebooks WHERE name = ?", (name,)).fetchone()
            is not None
        )

    if not taken(base):
        return base
    for n in range(1, 1000):
        suffix = " (Imported)" if n == 1 else f" (Imported {n})"
        candidate = base[: MAX_NAME_LENGTH - len(suffix)].rstrip() + suffix
        if not taken(candidate):
            return candidate
    raise ArchiveError("Could not find a free name for the imported notebook")


class _Importer:
    def __init__(
        self,
        conn: sqlite3.Connection,
        settings: Settings,
        reader: ArchiveReader,
        info: NotebookInfo,
        progress: Progress,
    ) -> None:
        self.conn, self.settings, self.reader, self.info = conn, settings, reader, info
        self.progress = progress
        self.maps: Maps = {t: {} for t in TABLES}
        self.staged: dict[int, Path] = {}
        self.moved: list[Path] = []
        self.staging = settings.archive_tmp_dir / f"import-{uuid.uuid4().hex}"
        self.notebook_id = 0
        self.reembed: dict[int, str] = {}  # old source id → why
        if not info.embeddings_compatible:
            why = f"made with {info.meta['embed_model']} ({info.meta['embed_dim']}-d)"
            self.reembed = dict.fromkeys(info.sources, why)
        for sid in info.invalid_embeddings:
            self.reembed[sid] = "some stored vectors were invalid"

    # --- staging -------------------------------------------------------------

    def stage_sources(self) -> None:
        self.staging.mkdir(parents=True, exist_ok=False)
        for sid, row in self.info.sources.items():
            dest = self.staging / f"{sid}.pdf"
            digest = hashlib.sha256()
            with create_exclusive(dest) as out:
                for chunk in self.reader.stream(
                    f"{self.info.prefix}/sources/{sid}.pdf"
                ):
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ArchiveError(
                    f"The PDF for “{row['original_filename']}” changed while reading"
                )
            self.staged[sid] = dest

    # --- rows ------------------------------------------------------------------

    def _insert(self, table: str, values: dict[str, Any]) -> int:
        cols = ", ".join(values)
        marks = ", ".join("?" * len(values))
        cur = self.conn.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values())
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def insert_all(self, name: str) -> None:
        meta, nb_prefix, m = self.info.meta, self.info.prefix, self.maps
        compatible = self.info.embeddings_compatible
        self.notebook_id = nb = self._insert(
            "notebooks",
            {
                "name": name,
                "description": meta["description"],
                "embed_model": meta["embed_model"] if compatible else EMBED_MODEL,
                "embed_dim": meta["embed_dim"] if compatible else EMBED_DIM,
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"],
            },
        )
        rows = self.reader.rows

        for r in rows(nb_prefix, "sources"):
            old = r.pop("id")
            if old in self.reembed:
                r.update(
                    status="failed",
                    stage="failed",
                    error=REEMBED_MESSAGE.format(why=self.reembed[old]),
                )
            new = self._insert("sources", {**r, "notebook_id": nb, "stored_path": ""})
            self.conn.execute(
                "UPDATE sources SET stored_path = ? WHERE id = ?",
                (
                    self.settings.source_file(nb, new)
                    .relative_to(self.settings.data_dir)
                    .as_posix(),
                    new,
                ),
            )
            m["sources"][old] = new
        for r in rows(nb_prefix, "pages"):
            old = r.pop("id")
            m["pages"][old] = self._insert(
                "pages", {**r, "source_id": m["sources"][r["source_id"]]}
            )
        for r in rows(nb_prefix, "chunks"):
            old = r.pop("id")
            embedding = None if r["source_id"] in self.reembed else r["embedding"]
            m["chunks"][old] = self._insert(
                "chunks",
                {
                    **r,
                    "notebook_id": nb,
                    "source_id": m["sources"][r["source_id"]],
                    "page_id": m["pages"][r["page_id"]],
                    "embedding": embedding,
                },
            )
        for r in rows(nb_prefix, "conversations"):
            old = r.pop("id")
            m["conversations"][old] = self._insert(
                "conversations", {**r, "notebook_id": nb}
            )
        for r in rows(nb_prefix, "messages"):
            old = r.pop("id")
            if r["status"] == "generating":  # exported mid-answer
                r.update(
                    status="interrupted",
                    error="Interrupted (exported while generating)",
                )
            m["messages"][old] = self._insert(
                "messages",
                {
                    **r,
                    "conversation_id": m["conversations"][r["conversation_id"]],
                    "reply_to": _soft(m, "messages", r["reply_to"], None),
                },
            )
        for r in rows(nb_prefix, "message_citations"):
            old = r.pop("id")
            m["message_citations"][old] = self._insert(
                "message_citations",
                {
                    **r,
                    "message_id": m["messages"][r["message_id"]],
                    "chunk_id": _soft(m, "chunks", r["chunk_id"], 0),
                    "source_id": _soft(m, "sources", r["source_id"], 0),
                },
            )
        for r in rows(nb_prefix, "memory_outcomes"):
            old = r.pop("id")
            m["memory_outcomes"][old] = self._insert(
                "memory_outcomes",
                {
                    **r,
                    "notebook_id": nb,
                    "message_id": m["messages"][r["message_id"]],
                },
            )
        for r in rows(nb_prefix, "annotations"):
            old = r.pop("id")
            m["annotations"][old] = self._insert(
                "annotations",
                {
                    **r,
                    "notebook_id": nb,
                    "source_id": m["sources"][r["source_id"]],
                },
            )
        for r in rows(nb_prefix, "annotation_rects"):
            old = r.pop("id")
            m["annotation_rects"][old] = self._insert(
                "annotation_rects",
                {
                    **r,
                    "annotation_id": m["annotations"][r["annotation_id"]],
                },
            )
        for r in rows(nb_prefix, "reading_state"):
            self._insert(
                "reading_state",
                {
                    **r,
                    "notebook_id": nb,
                    "source_id": m["sources"][r["source_id"]],
                },
            )
        for r in rows(nb_prefix, "notes"):
            old = r.pop("id")
            m["notes"][old] = self._insert("notes", {**r, "notebook_id": nb})
        link_keys: list[tuple[int, str, str, str | None]] = []
        for r in rows(nb_prefix, "note_links"):
            old = r.pop("id")
            new = self._insert(
                "note_links",
                {
                    **r,
                    "note_id": m["notes"][r["note_id"]],
                    "target_key": f"import:{old}",  # placeholder, final key below
                    "source_id": _soft(m, "sources", r["source_id"], None),
                    "chunk_id": _soft(m, "chunks", r["chunk_id"], None),
                    "message_id": _soft(m, "messages", r["message_id"], None),
                    "conversation_id": _soft(
                        m, "conversations", r["conversation_id"], None
                    ),
                    "annotation_id": _soft(m, "annotations", r["annotation_id"], None),
                    "details": _remap_json_ids(r["details"], m, "citations") or "{}",
                },
            )
            m["note_links"][old] = new
            link_keys.append((new, r["link_type"], r["target_key"], r["memory_key"]))
        for r in rows(nb_prefix, "message_memory_refs"):
            old = r.pop("id")
            m["message_memory_refs"][old] = self._insert(
                "message_memory_refs",
                {
                    **r,
                    "message_id": m["messages"][r["message_id"]],
                    "node_key": remap_key(r["node_key"], m),
                    "conversation_id": _soft(
                        m, "conversations", r["conversation_id"], None
                    ),
                    "sources": _remap_json_ids(r["sources"], m, None) or "[]",
                },
            )
        self._insert_study(nb)
        for new, link_type, target_key, memory_key in link_keys:
            self.conn.execute(
                "UPDATE note_links SET target_key = ?, memory_key = ? WHERE id = ?",
                (
                    remap_target(link_type, target_key, m),
                    remap_key(memory_key, m) if memory_key is not None else None,
                    new,
                ),
            )

    def _insert_study(self, nb: int) -> None:
        """Quizzes, attempts and study progress (format version 2+). Evidence
        snapshots keep their text; ids of sources, chunks, notes and chats
        are remapped, and become NULL when the item was not exported."""
        rows, prefix, m = self.reader.rows, self.info.prefix, self.maps
        for r in rows(prefix, "quizzes"):
            old = r.pop("id")
            for col in ("question_types", "material", "meta"):
                _check_json(r[col], f"quiz {old} {col}")
            material = json.loads(r["material"])
            if isinstance(material, dict) and isinstance(material.get("sources"), list):
                for item in material["sources"]:
                    if isinstance(item, dict) and isinstance(item.get("id"), int):
                        item["id"] = _soft(m, "sources", item["id"], None)
            if r["status"] == "generating":  # exported mid-generation
                r.update(
                    status="failed",
                    stage="failed",
                    error="Interrupted (exported while generating)",
                )
            m["quizzes"][old] = self._insert(
                "quizzes",
                {
                    **r,
                    "notebook_id": nb,
                    "material": json.dumps(material, ensure_ascii=False),
                },
            )
        for r in rows(prefix, "quiz_questions"):
            old = r.pop("id")
            _check_json(r["concepts"], f"quiz question {old} concepts")
            m["quiz_questions"][old] = self._insert(
                "quiz_questions", {**r, "quiz_id": m["quizzes"][r["quiz_id"]]}
            )
        for r in rows(prefix, "quiz_question_options"):
            old = r.pop("id")
            m["quiz_question_options"][old] = self._insert(
                "quiz_question_options",
                {**r, "question_id": m["quiz_questions"][r["question_id"]]},
            )
        for r in rows(prefix, "quiz_evidence"):
            old = r.pop("id")
            m["quiz_evidence"][old] = self._insert(
                "quiz_evidence",
                {
                    **r,
                    "question_id": m["quiz_questions"][r["question_id"]],
                    "source_id": _soft(m, "sources", r["source_id"], None),
                    "chunk_id": _soft(m, "chunks", r["chunk_id"], None),
                    "note_id": _soft(m, "notes", r["note_id"], None),
                    "message_id": _soft(m, "messages", r["message_id"], None),
                    "conversation_id": _soft(
                        m, "conversations", r["conversation_id"], None
                    ),
                },
            )
        for r in rows(prefix, "quiz_attempts"):
            old = r.pop("id")
            m["quiz_attempts"][old] = self._insert(
                "quiz_attempts",
                {
                    **r,
                    "quiz_id": m["quizzes"][r["quiz_id"]],
                    "parent_attempt_id": _soft(
                        m, "quiz_attempts", r["parent_attempt_id"], None
                    ),
                },
            )
        for r in rows(prefix, "quiz_attempt_items"):
            old = r.pop("id")
            if r["selected"] is not None:
                _check_json(r["selected"], f"quiz answer {old}")
            m["quiz_attempt_items"][old] = self._insert(
                "quiz_attempt_items",
                {
                    **r,
                    "attempt_id": m["quiz_attempts"][r["attempt_id"]],
                    "question_id": m["quiz_questions"][r["question_id"]],
                },
            )
        mismatched = self.conn.execute(
            "SELECT count(*) FROM quiz_attempt_items i JOIN quiz_attempts a "
            "ON a.id = i.attempt_id JOIN quiz_questions q ON q.id = i.question_id "
            "JOIN quizzes z ON z.id = a.quiz_id WHERE z.notebook_id = ? "
            "AND q.quiz_id != a.quiz_id",
            (nb,),
        ).fetchone()[0]
        if mismatched:
            raise ArchiveError("Quiz answers refer to questions of another quiz")
        for r in rows(prefix, "study_concept_stats"):
            old = r.pop("id")
            m["study_concept_stats"][old] = self._insert(
                "study_concept_stats", {**r, "notebook_id": nb}
            )

    def move_sources(self) -> None:
        for old, staged in self.staged.items():
            dest = self.settings.source_file(
                self.notebook_id, self.maps["sources"][old]
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise ArchiveError(
                    "A source file already exists at the import location"
                )
            replace_file(staged, dest)
            self.moved.append(dest)

    def cleanup_files(self, *, failed: bool) -> None:
        with contextlib.suppress(OSError):
            remove_tree(self.staging)
        if failed:
            for path in self.moved:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            if self.notebook_id:
                with contextlib.suppress(OSError):
                    remove_tree(self.settings.notebook_dir(self.notebook_id))


def import_notebook(
    conn: sqlite3.Connection,
    settings: Settings,
    reader: ArchiveReader,
    info: NotebookInfo,
    *,
    name: str | None = None,
    keep_name: bool = False,
    progress: Progress | None = None,
    _fail_at: str | None = None,  # tests: inject a failure at a stage
) -> ImportResult:
    """Import one validated notebook. ``name`` must be free if given;
    otherwise a collision gets "(Imported)" appended."""
    from litenotebook.index.faiss_store import FaissStore
    from litenotebook.memory.graph import rebuild as rebuild_graph
    from litenotebook.notebooks import NotebookError, delete_notebook

    started = time.perf_counter()
    stages: list[tuple[str, float]] = []
    say = progress or (lambda _stage: None)
    job = _Importer(conn, settings, reader, info, say)

    def stage(label: str) -> Callable[[], None]:
        say(label)
        t0 = time.perf_counter()
        if _fail_at == label:
            raise RuntimeError(f"injected failure at {label!r}")
        return lambda: stages.append((label, round(time.perf_counter() - t0, 3)))

    original = info.meta["name"]
    if name is not None:
        chosen = " ".join(name.split())
        if not chosen or len(chosen) > MAX_NAME_LENGTH:
            raise ArchiveError(f"Choose a name of 1–{MAX_NAME_LENGTH} characters")
        if conn.execute("SELECT 1 FROM notebooks WHERE name = ?", (chosen,)).fetchone():
            raise ArchiveError(f"A notebook named “{chosen}” already exists")
    else:
        chosen = original if keep_name else unique_name(conn, original)

    current = "copying sources"
    try:
        done = stage("Copying sources…")
        job.stage_sources()
        done()
        current = "writing to the database"
        done = stage("Writing notebook data…")
        if conn.in_transaction:  # never fold earlier implicit work into ours
            conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        try:
            job.insert_all(chosen)
            if _fail_at == "database":
                raise sqlite3.IntegrityError("injected database failure")
            current = "placing source files"
            job.move_sources()
            if _fail_at == "move":
                raise OSError("injected file move failure")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        done()
    except BaseException as e:
        job.cleanup_files(failed=True)
        if isinstance(e, KeyboardInterrupt | SystemExit):
            raise
        message = (
            str(e) if isinstance(e, ArchiveError) else type(e).__name__ + ": " + str(e)
        )
        raise ImportFailedError(current, message) from e

    try:
        current = "rebuilding the memory graph"
        done = stage("Rebuilding indexes…")
        rebuild_graph(conn, job.notebook_id)
        if _fail_at == "index":
            raise RuntimeError("injected index failure")
        current = "rebuilding the FAISS index"
        FaissStore(conn, settings).rebuild(job.notebook_id)
        done()
        current = "checking integrity"
        done = stage("Checking integrity…")
        report = notebook_report(conn, settings, job.notebook_id)
        if not report.ok:
            raise ArchiveError("; ".join(report.problems))
        done()
    except BaseException as e:
        job.cleanup_files(failed=False)
        with contextlib.suppress(NotebookError, sqlite3.Error, OSError):
            delete_notebook(conn, settings, job.notebook_id)
        if isinstance(e, KeyboardInterrupt | SystemExit):
            raise
        raise ImportFailedError(current, str(e)) from e
    job.cleanup_files(failed=False)
    say("Complete")
    warnings = []
    if job.reembed:
        warnings.append(
            f"{len(job.reembed)} source(s) need re-embedding before they can be "
            "searched; open Sources and use Retry"
        )
    return ImportResult(
        notebook_id=job.notebook_id,
        name=chosen,
        original_name=original,
        renamed=chosen != original,
        report=report,
        stages=stages,
        warnings=warnings,
        reembed_sources=len(job.reembed),
        seconds=round(time.perf_counter() - started, 3),
    )


def open_and_verify(
    path: Path, settings: Settings, conn: sqlite3.Connection
) -> tuple[ArchiveReader, VerifyReport]:
    reader = ArchiveReader(path, Limits.from_settings(settings))
    reader.__enter__()
    try:
        report = reader.validate(current_schema=schema_version(conn))
    except BaseException:
        reader.close()
        raise
    return reader, report


def import_archive(
    conn: sqlite3.Connection,
    settings: Settings,
    path: Path,
    *,
    name: str | None = None,
    progress: Progress | None = None,
) -> tuple[VerifyReport, list[ImportResult]]:
    """Import every notebook in a notebook export (one) or full backup (all),
    each under a collision-free name. Validation finishes before any write."""
    say = progress or (lambda _stage: None)
    say("Validating…")
    reader, report = open_and_verify(path, settings, conn)
    try:
        if name is not None and len(report.notebooks) != 1:
            raise ArchiveError("A new name can only be given for a single notebook")
        results: list[ImportResult] = []
        try:
            for info in report.notebooks:
                results.append(
                    import_notebook(
                        conn, settings, reader, info, name=name, progress=say
                    )
                )
        except BaseException:
            # all-or-nothing across notebooks too: undo the ones already imported
            from litenotebook.notebooks import NotebookError, delete_notebook

            for done in results:
                with contextlib.suppress(NotebookError, sqlite3.Error, OSError):
                    delete_notebook(conn, settings, done.notebook_id)
            raise
        return report, results
    finally:
        reader.close()


def settings_for(data_dir: Path, base: Settings) -> Settings:
    return replace(base, data_dir=data_dir)
