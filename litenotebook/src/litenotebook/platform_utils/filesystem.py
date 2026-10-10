"""Filesystem helpers with documented Linux / Windows behaviour.

- ``fsync_dir``: POSIX flushes a directory entry so a rename survives a
  crash; Windows cannot open a directory for fsync (NTFS journals metadata
  itself), so it is a no-op there. Best effort everywhere: a filesystem that
  refuses (some FUSE mounts) only loses the durability hint.
- ``create_exclusive``: a new file for binary writing. ``O_BINARY`` matters
  on Windows, where a descriptor opened without it translates newline bytes
  and silently corrupts PDFs and ZIPs.
- ``set_binary_mode``: puts an inherited pipe descriptor in binary mode on
  Windows (a no-op elsewhere), for the worker's protocol stream.
- ``replace_file``: ``os.replace``. Atomic for files on the same volume on
  both systems; on Windows a sharing violation (an indexer or antivirus has
  the target open for a moment) is retried briefly. Not atomic across volumes.
- ``rename_dir``: renames a directory to a path that must not exist. This is
  a single atomic rename on both systems, but Windows refuses while any file
  inside is open, so callers rename only directories nothing has open (the
  instance lock guarantees no server does). Sharing violations are retried.
- ``remove_tree``: ``shutil.rmtree`` that clears the read-only attribute,
  which Windows sets on some extracted files and which blocks deletion.
- ``is_within``: path containment by resolved components, never by string
  prefix (``C:\\data2`` is not inside ``C:\\data``).
- ``InstanceLock``: "a LiteNotebook process is changing this data directory".
  An OS lock (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on a byte range on
  Windows) on ``<data>/.instance.lock``. The OS drops it when the process
  exits or crashes, so a lock *file* left behind is never stale: the next
  process simply acquires it. The file also records who holds it (pid, time),
  for the "already in use" message.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from litenotebook.platform_utils.system import os_name

IS_WINDOWS = os_name() == "windows"
COPY_BUFFER_BYTES = 1 << 20  # streaming copies never hold more than 1 MiB
_RETRIES = 5


DISK_FULL_MESSAGE = (
    "Not enough disk space. Free some space on the drive that holds the "
    "LiteNotebook data folder and try again; nothing was changed."
)
_DISK_FULL_ERRNOS = {28, 122}  # ENOSPC, EDQUOT (Linux)
_DISK_FULL_WINERRORS = {39, 112}  # ERROR_HANDLE_DISK_FULL, ERROR_DISK_FULL


def is_disk_full(exc: BaseException) -> bool:
    """Is this the OS (or SQLite) saying the disk/quota is full?"""
    import sqlite3

    while exc is not None:
        if isinstance(exc, OSError) and (
            exc.errno in _DISK_FULL_ERRNOS
            or getattr(exc, "winerror", None) in _DISK_FULL_WINERRORS
        ):
            return True
        if isinstance(exc, sqlite3.OperationalError) and "disk is full" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return False


def fsync_dir(directory: Path) -> None:
    if IS_WINDOWS:
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # e.g. EINVAL on filesystems without directory fsync
    finally:
        os.close(fd)


def create_exclusive(path: Path) -> IO[bytes]:
    """Open a new file (it must not exist) for binary writing, owner-only where
    the OS supports permission bits."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "wb")


def set_binary_mode(fd: int) -> None:
    if IS_WINDOWS:
        import msvcrt

        msvcrt.setmode(fd, os.O_BINARY)  # type: ignore[attr-defined,unused-ignore]


def _retrying(action: Callable[[], None]) -> None:
    for attempt in range(_RETRIES):
        try:
            action()
            return
        except PermissionError:
            if not IS_WINDOWS or attempt == _RETRIES - 1:
                raise
            time.sleep(0.1 * (attempt + 1))  # transient sharing violation


def replace_file(src: Path, dst: Path) -> None:
    _retrying(lambda: os.replace(src, dst))


def rename_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Refusing to replace an existing path: {dst.name}")
    _retrying(lambda: os.rename(src, dst))


def _clear_readonly(func: Callable[..., Any], path: str, _exc: Any) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except PermissionError:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()
        return
    shutil.rmtree(path, onexc=_clear_readonly)


def is_within(path: Path, root: Path) -> bool:
    """``path`` resolves inside ``root`` (or is ``root``). Symlinks and ``..``
    are resolved first; comparison is by path components, not strings."""
    try:
        resolved = Path(path).resolve()
        base = Path(root).resolve()
    except (OSError, RuntimeError):
        return False
    return resolved == base or resolved.is_relative_to(base)


def copy_stream(
    src: IO[bytes],
    dst: IO[bytes],
    update: Callable[[bytes], None] | None = None,
    *,
    limit: int | None = None,
) -> int:
    """Copy in bounded chunks; optionally feed each chunk to ``update`` (a
    hash) and stop with ``ValueError`` once more than ``limit`` bytes arrive."""
    total = 0
    while chunk := src.read(COPY_BUFFER_BYTES):
        total += len(chunk)
        if limit is not None and total > limit:
            raise ValueError("stream is larger than its limit")
        if update is not None:
            update(chunk)
        dst.write(chunk)
    return total


class DataDirInUseError(RuntimeError):
    """Another LiteNotebook process is changing this data directory."""


class InstanceLock:
    """Advisory, non-blocking, crash-safe lock on a data directory."""

    FILENAME = ".instance.lock"
    _WIN_OFFSET = 1 << 20  # lock a byte beyond the info text so it stays readable

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / self.FILENAME
        self._file: IO[bytes] | None = None

    def acquire(
        self, purpose: str = "server", info: Mapping[str, Any] | None = None
    ) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600
        )
        f = os.fdopen(fd, "r+b")
        try:
            if IS_WINDOWS:
                import msvcrt

                f.seek(self._WIN_OFFSET)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined,unused-ignore]
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._file = f
        with contextlib.suppress(OSError):  # who holds it, for error messages
            f.seek(0)
            f.truncate()
            f.write(
                json.dumps(
                    {
                        **dict(info or {}),  # e.g. the app's loopback URL
                        "pid": os.getpid(),
                        "purpose": purpose,
                        "since": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                ).encode("utf-8")
            )
            f.flush()
        return True

    def release(self) -> None:
        f, self._file = self._file, None
        if f is None:
            return
        with contextlib.suppress(OSError):
            if IS_WINDOWS:
                import msvcrt

                f.seek(self._WIN_OFFSET)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined,unused-ignore]
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()

    def holder(self) -> dict[str, Any] | None:
        """What the lock file says about its holder (may be stale if unlocked)."""
        try:
            data = json.loads(self.path.read_bytes()[:4096].decode("utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def acquire_or_raise(
        self, purpose: str = "server", info: Mapping[str, Any] | None = None
    ) -> InstanceLock:
        if not self.acquire(purpose, info):
            who = self.holder() or {}
            detail = (
                f" (process {who['pid']}, {who.get('purpose', 'server')}, since "
                f"{who.get('since', '?')})"
                if isinstance(who.get("pid"), int)
                else ""
            )
            raise DataDirInUseError(
                "This LiteNotebook data folder is already in use by another "
                f"LiteNotebook process{detail}. Close it first, or use the running "
                "app's web interface."
            )
        return self

    @property
    def held(self) -> bool:
        return self._file is not None

    def __enter__(self) -> InstanceLock:
        return self.acquire_or_raise("command")

    def __exit__(self, *_exc: object) -> None:
        self.release()


def in_use(data_dir: Path) -> bool:
    """True if another process holds the data directory's instance lock."""
    lock = InstanceLock(data_dir)
    if lock.acquire("probe"):
        lock.release()
        return False
    return True
