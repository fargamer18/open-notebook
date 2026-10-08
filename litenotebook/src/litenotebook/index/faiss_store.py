"""One FAISS ``IndexIDMap2(IndexFlatIP(384))`` per notebook, derived from SQLite.

SQLite ``chunks.embedding`` is authoritative; vector ids are ``chunks.id``.
Index files are written atomically (temp file, fsync, ``os.replace``) next to a
``index.meta.json`` sidecar. ``check()`` detects a missing index, wrong model
or dimension, stale metadata, and any SQLite/FAISS id mismatch; ``sync()``
repairs incrementally and ``rebuild()`` recreates the index from SQLite.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from litenotebook.config import EMBED_DIM, EMBED_MODEL, Settings
from litenotebook.embed.embedder import FloatArray, from_blob, validate_vector

META_VERSION = 1

# faiss-cpu's bundled stubs omit IndexIDMap2.id_map and the numpy-array form
# of remove_ids() that the SWIG wrapper accepts, so index objects are untyped.
FaissIndex = Any


def _faiss() -> Any:
    """Import FAISS on first use: it maps ~30 MB of BLAS/OpenMP libraries, which
    processes that never search or index (e.g. the web server while browsing)
    should not pay for."""
    import faiss

    return faiss


_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


class IndexMismatchError(RuntimeError):
    pass


@dataclass
class IndexStatus:
    notebook_id: int
    sqlite_count: int
    index_count: int | None
    problems: list[str] = field(default_factory=list)
    needs_rebuild: bool = False  # structural problem: incremental sync impossible

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class SyncResult:
    added: int
    removed: int
    rebuilt: bool


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


def _fsync_dir(directory: Path) -> None:
    from litenotebook.platform_utils.filesystem import fsync_dir

    fsync_dir(directory)  # no-op on Windows, which cannot fsync a directory


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(tmp)
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        from litenotebook.platform_utils.filesystem import replace_file

        replace_file(tmp, path)  # retried on a Windows sharing violation
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def new_index(dim: int = EMBED_DIM) -> FaissIndex:
    return _faiss().IndexIDMap2(_faiss().IndexFlatIP(dim))


def index_ids(index: FaissIndex) -> npt.NDArray[np.int64]:
    return np.sort(_faiss().vector_to_array(index.id_map).astype(np.int64))


class FaissStore:
    def __init__(self, conn: sqlite3.Connection, settings: Settings) -> None:
        self.conn = conn
        self.settings = settings

    # --- paths & SQLite side ---------------------------------------------------

    def index_path(self, notebook_id: int) -> Path:
        return self.settings.faiss_index_path(notebook_id)

    def meta_path(self, notebook_id: int) -> Path:
        return self.index_path(notebook_id).with_name("index.meta.json")

    def _notebook_model(self, notebook_id: int) -> tuple[str, int]:
        row = self.conn.execute(
            "SELECT embed_model, embed_dim FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone()
        if row is None:
            raise IndexMismatchError(f"Notebook {notebook_id} not found")
        return row["embed_model"], row["embed_dim"]

    def sqlite_ids(self, notebook_id: int) -> npt.NDArray[np.int64]:
        rows = self.conn.execute(
            "SELECT id FROM chunks WHERE notebook_id = ? AND embedding IS NOT NULL "
            "ORDER BY id",
            (notebook_id,),
        ).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64)

    def _vectors(
        self, notebook_id: int, ids: npt.NDArray[np.int64] | None = None
    ) -> tuple[npt.NDArray[np.int64], FloatArray]:
        sql = (
            "SELECT id, embedding FROM chunks WHERE notebook_id = ? "
            "AND embedding IS NOT NULL"
        )
        params: list[Any] = [notebook_id]
        if ids is not None:
            sql += " AND id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps([int(i) for i in ids]))
        rows = self.conn.execute(sql + " ORDER BY id", params).fetchall()
        out_ids = np.array([r["id"] for r in rows], dtype=np.int64)
        vectors = np.zeros((len(rows), EMBED_DIM), dtype=np.float32)
        for i, row in enumerate(rows):
            vectors[i] = from_blob(row["embedding"])
        return out_ids, vectors

    # --- reading & checking ----------------------------------------------------

    def _read(
        self, notebook_id: int
    ) -> tuple[FaissIndex | None, dict[str, Any] | None]:
        index_path, meta_path = (
            self.index_path(notebook_id),
            self.meta_path(notebook_id),
        )
        index = _faiss().read_index(str(index_path)) if index_path.exists() else None
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = None
        return index, meta

    def check(self, notebook_id: int) -> IndexStatus:
        model, dim = self._notebook_model(notebook_id)
        expected = self.sqlite_ids(notebook_id)
        status = IndexStatus(notebook_id, sqlite_count=len(expected), index_count=None)
        problems = status.problems

        if model != EMBED_MODEL or dim != EMBED_DIM:
            problems.append(
                f"notebook uses {model} ({dim}-d) but this build embeds with "
                f"{EMBED_MODEL} ({EMBED_DIM}-d)"
            )
        try:
            index, meta = self._read(notebook_id)
        except RuntimeError as e:  # FAISS raises RuntimeError on corrupt files
            problems.append(f"index file unreadable: {e}")
            status.needs_rebuild = True
            return status
        if index is None:
            if len(expected) == 0:
                return status  # nothing embedded yet: no index is the correct state
            problems.append("index file missing")
            status.needs_rebuild = True
            return status

        status.index_count = int(index.ntotal)
        if not isinstance(index, _faiss().IndexIDMap2):
            problems.append(f"index type is {type(index).__name__}, not IndexIDMap2")
            status.needs_rebuild = True
        if index.d != dim:
            problems.append(f"index dimension {index.d} != notebook dimension {dim}")
            status.needs_rebuild = True
        if meta is None:
            problems.append("index metadata missing or unreadable")
        else:
            if meta.get("embed_model") != model:
                problems.append(
                    f"index built with {meta.get('embed_model')!r}, "
                    f"notebook uses {model!r}"
                )
                status.needs_rebuild = True
            if meta.get("embed_dim") != dim:
                problems.append(
                    f"index metadata dimension {meta.get('embed_dim')} != {dim}"
                )
                status.needs_rebuild = True
            if meta.get("ntotal") != index.ntotal:
                problems.append("index metadata is stale (count differs from index)")
        if not status.needs_rebuild:
            actual = index_ids(index)
            if len(actual) != len(expected):
                problems.append(
                    f"count mismatch: SQLite has {len(expected)} embedded chunks, "
                    f"FAISS has {len(actual)}"
                )
            elif not np.array_equal(actual, expected):
                problems.append("id mismatch between SQLite and FAISS")
        return status

    # --- writing ---------------------------------------------------------------

    def _save(self, notebook_id: int, index: FaissIndex) -> None:
        model, dim = self._notebook_model(notebook_id)
        meta = {
            "version": META_VERSION,
            "embed_model": model,
            "embed_dim": dim,
            "ntotal": int(index.ntotal),
            "saved_at": datetime.now(UTC).isoformat(),
        }
        _atomic_write(
            self.index_path(notebook_id), lambda p: _faiss().write_index(index, str(p))
        )

        def write_meta(path: Path) -> None:
            path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        _atomic_write(self.meta_path(notebook_id), write_meta)

    def rebuild(self, notebook_id: int) -> FaissIndex:
        """Recreate the notebook's index from SQLite embedding BLOBs."""
        model, dim = self._notebook_model(notebook_id)
        if model != EMBED_MODEL or dim != EMBED_DIM:
            raise IndexMismatchError(
                f"Notebook embeddings come from {model} ({dim}-d); re-embedding is "
                f"required, a FAISS rebuild cannot fix this"
            )
        with _lock_for(self.index_path(notebook_id)):
            ids, vectors = self._vectors(notebook_id)
            index = new_index(dim)
            if len(ids):
                index.add_with_ids(vectors, ids)
            self._save(notebook_id, index)
            return index

    def sync(self, notebook_id: int) -> SyncResult:
        """Bring FAISS in line with SQLite: add missing ids, drop extra ones."""
        status = self.check(notebook_id)
        if status.ok:
            return SyncResult(0, 0, rebuilt=False)
        if status.needs_rebuild:
            index = self.rebuild(notebook_id)
            return SyncResult(int(index.ntotal), 0, rebuilt=True)
        with _lock_for(self.index_path(notebook_id)):
            current, _ = self._read(notebook_id)
            assert current is not None
            expected, actual = self.sqlite_ids(notebook_id), index_ids(current)
            extra = np.setdiff1d(actual, expected)
            missing = np.setdiff1d(expected, actual)
            if len(extra):
                current.remove_ids(extra)
            if len(missing):
                ids, vectors = self._vectors(notebook_id, missing)
                current.add_with_ids(vectors, ids)
            self._save(notebook_id, current)
            return SyncResult(len(missing), len(extra), rebuilt=False)

    # --- querying --------------------------------------------------------------

    def load(self, notebook_id: int, *, repair: bool = True) -> FaissIndex:
        """Load a validated index; repair it from SQLite (or raise) if inconsistent."""
        status = self.check(notebook_id)
        if not status.ok:
            if not repair:
                raise IndexMismatchError("; ".join(status.problems))
            self.sync(notebook_id)
        index, _ = self._read(notebook_id)
        if index is None:  # empty notebook: nothing to index yet
            return new_index(self._notebook_model(notebook_id)[1])
        return index

    def search(
        self, notebook_id: int, query: FloatArray, k: int
    ) -> list[tuple[int, float]]:
        """Top-k ``(chunk_id, inner_product)`` pairs for a normalized query vector."""
        validate_vector(query)
        index = self.load(notebook_id)
        if index.ntotal == 0 or k <= 0:
            return []
        scores, ids = index.search(query.reshape(1, -1), min(k, int(index.ntotal)))
        return [
            (int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0
        ]
