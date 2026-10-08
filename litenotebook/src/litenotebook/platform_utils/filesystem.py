"""Filesystem helpers with documented Linux / Windows behaviour.

- ``fsync_dir``: POSIX flushes the directory entry so a rename survives a
  crash; Windows cannot open a directory for fsync (and NTFS journals
  metadata), so it is a no-op there.
- ``create_exclusive``: creates a new file for binary writing. ``O_BINARY``
  matters on Windows, where a descriptor opened without it translates
  newline bytes and silently corrupts PDFs and ZIPs.
- ``replace_file``: ``os.replace`` (atomic on both systems for files on the
  same volume). On Windows another process (an indexer or antivirus) may hold
  the target open for a moment, so a sharing violation is retried briefly.
- ``rename_dir``: renames a directory to a path that must not exist. Windows
  refuses while any file inside is open, so callers rename only directories
  that no LiteNotebook process is using (see ``InstanceLock``).
- ``remove_tree``: ``shutil.rmtree`` that clears the read-only attribute,
  which Windows sets on some extracted files and which blocks deletion.
- ``InstanceLock``: an advisory lock on a file inside the data directory,
  held by a running server (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on
  Windows). Released automatically if the process dies.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

IS_WINDOWS = sys.platform == "win32"
COPY_BUFFER_BYTES = 1 << 20  # streaming copies never hold more than 1 MiB


def fsync_dir(directory: Path) -> None:
    if IS_WINDOWS:
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_exclusive(path: Path) -> IO[bytes]:
    """Open a new file (it must not exist) for binary writing, owner-only where
    the OS supports permission bits."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, "wb")


def replace_file(src: Path, dst: Path, *, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if not IS_WINDOWS or attempt == attempts - 1:
                raise
            time.sleep(0.1 * (attempt + 1))  # transient sharing violation


def rename_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Refusing to replace an existing path: {dst.name}")
    os.rename(src, dst)


def _clear_readonly(func: Callable[..., Any], path: str, _exc: Any) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path, onexc=_clear_readonly)


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


class InstanceLock:
    """Advisory, non-blocking lock meaning "a LiteNotebook server uses this
    data directory". ``acquire()`` returns False when someone else holds it."""

    FILENAME = ".instance.lock"

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / self.FILENAME
        self._file: IO[bytes] | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+b")  # noqa: SIM115 - held until release()
        try:
            if IS_WINDOWS:
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._file = f
        return True

    def release(self) -> None:
        f, self._file = self._file, None
        if f is None:
            return
        with contextlib.suppress(OSError):
            if IS_WINDOWS:
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()

    @property
    def held(self) -> bool:
        return self._file is not None

    def __enter__(self) -> InstanceLock:
        if not self.acquire():
            raise BlockingIOError("LiteNotebook is running with this data directory")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def in_use(data_dir: Path) -> bool:
    """True if a server holds the data directory's instance lock."""
    lock = InstanceLock(data_dir)
    if lock.acquire():
        lock.release()
        return False
    return True
