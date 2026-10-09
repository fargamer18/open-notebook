"""Open and fully validate an archive without changing anything.

Archives are hostile input. Nothing is ever extracted by its archive path:
entries are read through ``zipfile`` streams into destinations the caller
chooses, and every size limit is enforced on the bytes actually produced
(header sizes can lie). Validation order:

1. archive file size, ZIP central directory, entry count;
2. every entry name (portable-path rules, then an exact allowlist), no
   directories/symlinks/encryption/odd compression, no exact or
   case-insensitive duplicates, declared and total sizes;
3. manifest: format name, format version (newer → refused with a clear
   message), kind, schema version, notebook list;
4. checksums.json against the manifest, then a streamed SHA-256 of every
   entry against checksums.json;
5. every JSONL row: allowed columns only, value types, unique ids, parent
   references inside the archive, embedding BLOBs, and each source row's PDF
   (present, same size and SHA-256, ``%PDF-`` header).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import stat
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import numpy as np

from litenotebook.archive.format import (
    BLOB,
    CHECKSUMS,
    ENTRY_PATTERNS,
    FORMAT_NAME,
    FORMAT_VERSION,
    INT,
    KINDS,
    MANIFEST,
    NEWER_VERSION_MESSAGE,
    NOTEBOOK,
    REAL,
    SETTINGS,
    TABLES,
    TEXT,
    ArchiveError,
    Limits,
    UnsupportedArchiveError,
    tables_for,
)
from litenotebook.config import EMBED_DIM, EMBED_MODEL
from litenotebook.platform_utils.filesystem import COPY_BUFFER_BYTES
from litenotebook.platform_utils.paths import archive_name_problem, portable_key

ALLOWED_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
_HEX64 = frozenset("0123456789abcdef")

# Hard references: the parent row must be in the same archive.
PARENTS: dict[str, list[tuple[str, str]]] = {
    "pages": [("source_id", "sources")],
    "chunks": [("source_id", "sources"), ("page_id", "pages")],
    "messages": [("conversation_id", "conversations"), ("reply_to", "messages")],
    "message_citations": [("message_id", "messages")],
    "memory_outcomes": [("message_id", "messages")],
    "annotations": [("source_id", "sources")],
    "annotation_rects": [("annotation_id", "annotations")],
    "reading_state": [("source_id", "sources")],
    "note_links": [("note_id", "notes")],
    "message_memory_refs": [("message_id", "messages")],
    "quiz_questions": [("quiz_id", "quizzes")],
    "quiz_question_options": [("question_id", "quiz_questions")],
    "quiz_evidence": [("question_id", "quiz_questions")],
    "quiz_attempts": [("quiz_id", "quizzes"), ("parent_attempt_id", "quiz_attempts")],
    "quiz_attempt_items": [
        ("attempt_id", "quiz_attempts"),
        ("question_id", "quiz_questions"),
    ],
}


@dataclass
class NotebookInfo:
    original_id: int
    prefix: str
    meta: dict[str, Any]
    counts: dict[str, int] = field(default_factory=dict)
    sources: dict[int, dict[str, Any]] = field(default_factory=dict)  # id → row
    invalid_embeddings: dict[int, int] = field(default_factory=dict)  # source → n
    embeddings_compatible: bool = True


@dataclass
class VerifyReport:
    path: Path
    size: int
    sha256: str
    manifest: dict[str, Any]
    notebooks: list[NotebookInfo]
    entries: int
    expanded_bytes: int
    seconds: float
    settings: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return str(self.manifest.get("kind", ""))


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode) if mode else False


def _decode_blob(value: str, where: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ArchiveError(f"{where}: embedding is not valid base64") from e


def embedding_problem(blob: bytes, dim: int = EMBED_DIM) -> str | None:
    """Why a stored embedding cannot be trusted, or None if it is usable."""
    if len(blob) != dim * 4:
        return f"{len(blob)} bytes instead of {dim * 4}"
    vector = np.frombuffer(blob, dtype="<f4")
    if not np.all(np.isfinite(vector)):
        return "NaN or infinite values"
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > 1e-3:
        return f"not L2-normalized (norm {norm:.4f})"
    return None


class ArchiveReader:
    """A validated, read-only view of an archive. Use as a context manager."""

    def __init__(self, path: Path, limits: Limits) -> None:
        self.path = Path(path)
        self.limits = limits
        self.zf: zipfile.ZipFile | None = None
        self.infos: dict[str, zipfile.ZipInfo] = {}
        self.manifest: dict[str, Any] = {}
        self.version = FORMAT_VERSION  # set from the manifest by validate()
        self.checksums: dict[str, str] = {}

    # --- lifecycle -----------------------------------------------------------

    def __enter__(self) -> ArchiveReader:
        self._open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()
            self.zf = None

    def _open(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError as e:
            raise ArchiveError("The archive file could not be read") from e
        if size > self.limits.max_archive_bytes:
            raise ArchiveError(
                f"The archive is larger than the "
                f"{self.limits.max_archive_bytes // 2**20} MiB limit"
            )
        try:
            self.zf = zipfile.ZipFile(self.path)
            infos = self.zf.infolist()
        except (
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
            OSError,
            ValueError,
            NotImplementedError,
            EOFError,
        ) as e:
            self.close()
            raise ArchiveError("This is not a valid ZIP archive") from e
        try:
            self._check_entries(infos)
        except BaseException:
            self.close()
            raise

    # --- structure -----------------------------------------------------------

    def _check_entries(self, infos: list[zipfile.ZipInfo]) -> None:
        lim = self.limits
        if len(infos) > lim.max_files:
            raise ArchiveError(f"The archive has more than {lim.max_files} entries")
        seen: dict[str, str] = {}
        expanded = 0
        for info in infos:
            name = info.filename
            for candidate in (info.orig_filename, name):  # raw name too (Windows)
                problem = archive_name_problem(candidate)
                if problem:
                    raise ArchiveError(f"Unsafe archive entry ({problem}): {name!r}")
            if info.is_dir():
                raise ArchiveError(f"Unexpected directory entry: {name!r}")
            if _is_symlink(info):
                raise ArchiveError(f"Symbolic links are not allowed: {name!r}")
            if info.flag_bits & 0x1:
                raise ArchiveError(f"Encrypted entries are not supported: {name!r}")
            if info.compress_type not in ALLOWED_COMPRESSION:
                raise ArchiveError(f"Unsupported compression in {name!r}")
            key = portable_key(name)
            if key in seen:
                raise ArchiveError(
                    f"Duplicate entry {name!r} (same as {seen[key]!r} on a "
                    "case-insensitive filesystem)"
                )
            seen[key] = name
            if not any(p.match(name) for p in ENTRY_PATTERNS.values()):
                raise ArchiveError(f"Unexpected entry in archive: {name!r}")
            if info.file_size > lim.max_entry_bytes:
                raise ArchiveError(f"Entry {name!r} is larger than the per-file limit")
            expanded += info.file_size
            if expanded > lim.max_expanded_bytes:
                raise ArchiveError("The archive expands beyond the size limit")
            self.infos[name] = info
        self.expanded_bytes = expanded
        for required in (MANIFEST, CHECKSUMS):
            if required not in self.infos:
                raise ArchiveError(f"Not a LiteNotebook archive: {required} is missing")

    # --- bounded reads -----------------------------------------------------------

    def open_entry(self, name: str) -> IO[bytes]:
        assert self.zf is not None
        return self.zf.open(self.infos[name])

    def stream(self, name: str, limit: int | None = None) -> Iterator[bytes]:
        """Entry bytes in bounded chunks; never more than declared or ``limit``."""
        info = self.infos[name]
        cap = min(info.file_size, limit if limit is not None else info.file_size)
        if info.file_size > cap:
            raise ArchiveError(f"Entry {name!r} is too large")
        total = 0
        try:
            with self.open_entry(name) as f:
                while chunk := f.read(COPY_BUFFER_BYTES):
                    total += len(chunk)
                    if total > info.file_size:
                        raise ArchiveError(f"Entry {name!r} is larger than declared")
                    yield chunk
        except (zipfile.BadZipFile, OSError, EOFError, ValueError) as e:
            if isinstance(e, ArchiveError):
                raise
            raise ArchiveError(f"Entry {name!r} is corrupt") from e
        if total != info.file_size:
            raise ArchiveError(f"Entry {name!r} is truncated")

    def read_json(self, name: str) -> Any:
        data = b"".join(self.stream(name, self.limits.max_json_bytes))
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ArchiveError(f"{name} is not valid JSON") from e

    def sha256(self, name: str) -> str:
        digest = hashlib.sha256()
        for chunk in self.stream(name):
            digest.update(chunk)
        return digest.hexdigest()

    def iter_lines(self, name: str) -> Iterator[bytes]:
        buffer = b""
        for chunk in self.stream(name):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line:
                    yield line
            if len(buffer) > self.limits.max_line_bytes:
                raise ArchiveError(f"{name}: a row is larger than the limit")
        if buffer.strip():
            yield buffer

    def rows(self, prefix: str, table: str) -> Iterator[dict[str, Any]]:
        """Validated rows: exactly the allowed columns with the right types.
        BLOB columns come back as bytes."""
        name = f"{prefix}/tables/{table}.jsonl"
        if table not in tables_for(self.version):
            return  # an older archive: this table did not exist yet
        columns = TABLES[table]
        expected = {c.name for c in columns}
        for n, raw in enumerate(self.iter_lines(name), 1):
            where = f"{name} row {n}"
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise ArchiveError(f"{where} is not valid JSON") from e
            if not isinstance(row, dict) or set(row) != expected:
                unknown = sorted(set(row) - expected) if isinstance(row, dict) else []
                hint = f" (unknown: {', '.join(unknown)})" if unknown else ""
                raise ArchiveError(f"{where} has unexpected columns{hint}")
            for col in columns:
                row[col.name] = _check_value(
                    row[col.name], col.kind, col.nullable, f"{where}.{col.name}"
                )
            yield row

    # --- full validation -----------------------------------------------------------

    def validate(self, *, current_schema: int) -> VerifyReport:
        started = time.perf_counter()
        manifest = self.read_json(MANIFEST)
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT_NAME:
            raise ArchiveError("Not a LiteNotebook archive (unknown manifest format)")
        version = manifest.get("export_format_version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ArchiveError("The archive manifest has an invalid format version")
        if version > FORMAT_VERSION:
            raise UnsupportedArchiveError(NEWER_VERSION_MESSAGE)
        self.version = version
        schema = manifest.get("schema_version")
        if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
            raise ArchiveError("The archive manifest has an invalid schema version")
        if schema > current_schema:
            raise UnsupportedArchiveError(NEWER_VERSION_MESSAGE)
        kind = manifest.get("kind")
        if kind not in KINDS:
            raise ArchiveError("The archive manifest has an unknown kind")
        self.manifest = manifest

        raw_checksums = b"".join(self.stream(CHECKSUMS, self.limits.max_json_bytes))
        if hashlib.sha256(raw_checksums).hexdigest() != manifest.get(
            "checksums_sha256"
        ):
            raise ArchiveError(
                "checksums.json does not match the manifest (corrupt archive)"
            )
        try:
            checksums = json.loads(raw_checksums.decode("utf-8")).get("files")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as e:
            raise ArchiveError("checksums.json is not valid") from e
        if not isinstance(checksums, dict) or not all(
            isinstance(v, str) and len(v) == 64 and set(v) <= _HEX64
            for v in checksums.values()
        ):
            raise ArchiveError("checksums.json is not valid")
        listed = set(checksums)
        present = set(self.infos) - {MANIFEST, CHECKSUMS}
        if listed != present:
            missing, extra = sorted(listed - present), sorted(present - listed)
            raise ArchiveError(
                "The archive's file list does not match its checksums "
                f"(missing: {missing[:3]}, unlisted: {extra[:3]})"
            )
        for name in sorted(present):
            if self.sha256(name) != checksums[name]:
                raise ArchiveError(f"Checksum mismatch for {name} (corrupt archive)")
        self.checksums = checksums

        notebooks = self._validate_notebooks(manifest, kind)
        settings_rows: list[dict[str, str]] = []
        if SETTINGS in self.infos:
            if kind != "backup":
                raise ArchiveError("settings.json is only allowed in full backups")
            data = self.read_json(SETTINGS)
            rows = data.get("settings") if isinstance(data, dict) else None
            if not isinstance(rows, list) or not all(
                isinstance(r, dict)
                and set(r) == {"key", "value"}
                and isinstance(r["key"], str)
                and isinstance(r["value"], str)
                for r in rows
            ):
                raise ArchiveError("settings.json is not valid")
            settings_rows = rows
        warnings = []
        for nb in notebooks:
            if not nb.embeddings_compatible:
                warnings.append(
                    f"“{nb.meta['name']}” was embedded with {nb.meta['embed_model']} "
                    f"({nb.meta['embed_dim']}-d); its sources will need re-embedding"
                )
            bad = sum(nb.invalid_embeddings.values())
            if bad:
                warnings.append(
                    f"“{nb.meta['name']}”: {bad} stored embedding(s) are invalid; "
                    f"{len(nb.invalid_embeddings)} source(s) will need re-embedding"
                )
        digest = hashlib.sha256()
        with open(self.path, "rb") as f:
            while chunk := f.read(COPY_BUFFER_BYTES):
                digest.update(chunk)
        return VerifyReport(
            path=self.path,
            size=self.path.stat().st_size,
            sha256=digest.hexdigest(),
            manifest=manifest,
            notebooks=notebooks,
            entries=len(self.infos),
            expanded_bytes=self.expanded_bytes,
            seconds=round(time.perf_counter() - started, 3),
            settings=settings_rows,
            warnings=warnings,
        )

    def _validate_notebooks(
        self, manifest: dict[str, Any], kind: str
    ) -> list[NotebookInfo]:
        listed = manifest.get("notebooks")
        if not isinstance(listed, list) or not all(isinstance(n, dict) for n in listed):
            raise ArchiveError("The archive manifest has no valid notebook list")
        dirs: dict[int, set[str]] = {}
        for name in self.infos:
            for pattern in ("notebook", "table", "source"):
                m = ENTRY_PATTERNS[pattern].match(name)
                if m:
                    dirs.setdefault(int(m.group("nb")), set()).add(name)
        ids = []
        for n in listed:
            nb_id = n.get("id")
            if not isinstance(nb_id, int) or isinstance(nb_id, bool) or nb_id < 1:
                raise ArchiveError("The archive manifest lists an invalid notebook")
            ids.append(nb_id)
        if len(set(ids)) != len(ids) or set(ids) != set(dirs):
            raise ArchiveError(
                "The manifest's notebooks do not match the archive contents"
            )
        if kind == "notebook" and len(ids) != 1:
            raise ArchiveError("A notebook export must contain exactly one notebook")
        return [self._validate_notebook(nb_id) for nb_id in ids]

    def _validate_notebook(self, nb_id: int) -> NotebookInfo:
        prefix = f"notebooks/{nb_id}"
        meta_name = f"{prefix}/notebook.json"
        if meta_name not in self.infos:
            raise ArchiveError(f"{meta_name} is missing")
        meta = self.read_json(meta_name)
        expected = {c.name for c in NOTEBOOK}
        if not isinstance(meta, dict) or set(meta) != expected:
            raise ArchiveError(f"{meta_name} has unexpected fields")
        for col in NOTEBOOK:
            meta[col.name] = _check_value(
                meta[col.name], col.kind, col.nullable, f"{meta_name}.{col.name}"
            )
        if not meta["name"].strip():
            raise ArchiveError(f"{meta_name}: the notebook name is empty")
        expected_tables = tables_for(self.version)
        for table in expected_tables:
            if f"{prefix}/tables/{table}.jsonl" not in self.infos:
                raise ArchiveError(f"{prefix}/tables/{table}.jsonl is missing")
        for name in self.infos:
            m = ENTRY_PATTERNS["table"].match(name)
            if (
                m
                and m.group("nb") == str(nb_id)
                and m.group("table") not in expected_tables
            ):
                raise ArchiveError(f"Unknown table in archive: {name!r}")

        info = NotebookInfo(nb_id, prefix, meta)
        info.embeddings_compatible = (
            meta["embed_model"] == EMBED_MODEL and meta["embed_dim"] == EMBED_DIM
        )
        ids: dict[str, set[int]] = {}
        for table in expected_tables:
            key = "source_id" if table == "reading_state" else "id"
            seen: set[int] = set()
            parents = PARENTS.get(table, [])
            n = 0
            for row in self.rows(prefix, table):
                n += 1
                if row[key] in seen:
                    raise ArchiveError(f"{prefix}/tables/{table}.jsonl: duplicate id")
                seen.add(row[key])
                for ref, parent in parents:
                    value = row[ref]
                    if value is not None and value not in ids.get(
                        parent, seen if parent == table else set()
                    ):
                        raise ArchiveError(
                            f"{prefix}/tables/{table}.jsonl: {ref} {value} refers to a "
                            f"row that is not in the archive"
                        )
                if table == "sources":
                    info.sources[row["id"]] = row
                    if row["status"] not in ("ready", "failed"):
                        raise ArchiveError(
                            f"Source “{row['original_filename']}” has status "
                            f"{row['status']!r}; only finished sources can be imported"
                        )
                if table == "chunks" and row["embedding"] is not None:
                    problem = embedding_problem(row["embedding"], meta["embed_dim"])
                    if problem and info.embeddings_compatible:
                        sid = row["source_id"]
                        info.invalid_embeddings[sid] = (
                            info.invalid_embeddings.get(sid, 0) + 1
                        )
            ids[table] = seen
            info.counts[table] = n
        self._validate_sources(info)
        return info

    def _validate_sources(self, info: NotebookInfo) -> None:
        pdfs = {
            int(m.group("sid"))
            for name in self.infos
            if (m := ENTRY_PATTERNS["source"].match(name))
            and m.group("nb") == str(info.original_id)
        }
        if pdfs != set(info.sources):
            raise ArchiveError(
                f"{info.prefix}: source PDFs do not match the source rows"
            )
        for sid, row in info.sources.items():
            name = f"{info.prefix}/sources/{sid}.pdf"
            entry = self.infos[name]
            if (
                entry.file_size != row["size_bytes"]
                or self.checksums[name] != row["sha256"]
            ):
                raise ArchiveError(
                    f"The PDF for “{row['original_filename']}” does not match its "
                    "recorded size or SHA-256 (corrupt archive)"
                )
            with self.open_entry(name) as f:
                if f.read(5) != b"%PDF-":
                    raise ArchiveError(
                        f"The file for “{row['original_filename']}” is not a PDF"
                    )
        info.counts["source_bytes"] = sum(
            r["size_bytes"] for r in info.sources.values()
        )


def _check_value(value: Any, kind: str, nullable: bool, where: str) -> Any:
    if value is None:
        if nullable:
            return None
        raise ArchiveError(f"{where} must not be empty")
    if kind == INT:
        if not isinstance(value, int) or isinstance(value, bool) or abs(value) > 2**62:
            raise ArchiveError(f"{where} must be an integer")
    elif kind == TEXT:
        if not isinstance(value, str) or "\x00" in value:
            raise ArchiveError(f"{where} must be text")
    elif kind == REAL:
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ArchiveError(f"{where} must be a number")
        value = float(value)
    elif kind == BLOB:
        if not isinstance(value, str):
            raise ArchiveError(f"{where} must be base64 text")
        value = _decode_blob(value, where)
    return value


def verify_archive(path: Path, limits: Limits, *, current_schema: int) -> VerifyReport:
    """Validate everything; never writes anywhere."""
    with ArchiveReader(path, limits) as reader:
        return reader.validate(current_schema=current_schema)
