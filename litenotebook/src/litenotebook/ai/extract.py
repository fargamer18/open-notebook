"""Safe extraction of downloaded runtime archives (zip / tar.gz / tar.bz2).

The same rules as notebook backups (Phase 10): every name must be a portable
relative path (no ``..``, absolute, drive, UNC, backslash, control, reserved
Windows names), no duplicates or case collisions, sizes counted on the bytes
actually read, and only regular files and directories are written — into a
fresh staging directory chosen by LiteNotebook.

Symbolic and hard links inside the archive (Linux builds ship ``libfoo.so ->
libfoo.so.0``) are never created as links: once everything is extracted, a
link whose target is another regular file *inside the archive* is
materialised as a copy; a link pointing anywhere else rejects the archive.
Devices, FIFOs and other special entries are rejected.
"""

from __future__ import annotations

import posixpath
import shutil
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from litenotebook.platform_utils.filesystem import (
    COPY_BUFFER_BYTES,
    create_exclusive,
    is_within,
)
from litenotebook.platform_utils.paths import archive_name_problem, portable_key
from litenotebook.platform_utils.system import os_name


class UnsafeArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    max_files: int = 10_000
    max_total_bytes: int = 8 * 1024**3
    max_entry_bytes: int = 4 * 1024**3


DEFAULT_LIMITS = Limits()


def _clean(name: str) -> str:
    name = name[2:] if name.startswith("./") else name
    name = name.rstrip("/")
    problem = archive_name_problem(name)
    if problem:
        raise UnsafeArchiveError(
            f"Unsafe path in runtime archive ({problem}): {name!r}"
        )
    return name


def _write(src: IO[bytes], target: Path, budget: list[int], limits: Limits) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with create_exclusive(target) as out:
        while chunk := src.read(COPY_BUFFER_BYTES):
            written += len(chunk)
            budget[0] += len(chunk)
            if written > limits.max_entry_bytes:
                raise UnsafeArchiveError("An entry in the runtime archive is too large")
            if budget[0] > limits.max_total_bytes:
                raise UnsafeArchiveError(
                    "The runtime archive expands beyond the size limit"
                )
            out.write(chunk)
    return written


def _executable_bit(target: Path, mode: int) -> None:
    if os_name() == "windows":
        return
    target.chmod(0o755 if mode & 0o111 else 0o644)


class _Seen:
    def __init__(self, limits: Limits) -> None:
        self.keys: dict[str, str] = {}
        self.limits = limits

    def add(self, name: str) -> None:
        key = portable_key(name)
        if key in self.keys:
            raise UnsafeArchiveError(f"Duplicate entry in runtime archive: {name!r}")
        self.keys[key] = name
        if len(self.keys) > self.limits.max_files:
            raise UnsafeArchiveError("The runtime archive has too many entries")


def _materialise_links(dest: Path, links: list[tuple[str, str]]) -> None:
    """Replace in-archive links by copies of their (regular-file) targets."""
    pending = dict(links)
    for _ in range(len(pending) + 1):  # links to links resolve over passes
        progress = False
        for name, target in list(pending.items()):
            source = dest / target
            if source.is_file() and not source.is_symlink():
                out = dest / name
                if out.exists() or out.is_symlink():
                    raise UnsafeArchiveError(
                        f"Runtime archive link collides with another entry: {name!r}"
                    )
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, out)
                shutil.copymode(source, out)
                del pending[name]
                progress = True
        if not progress:
            break
    if pending:
        raise UnsafeArchiveError(
            f"Runtime archive link points outside the archive: {next(iter(pending))!r}"
        )


def _link_target(name: str, target: str) -> str:
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
    if target.startswith("/") or joined.startswith("../") or joined == "..":
        raise UnsafeArchiveError(f"Runtime archive link escapes the archive: {name!r}")
    return _clean(joined)


def extract_zip(
    archive: Path, dest: Path, limits: Limits = DEFAULT_LIMITS
) -> list[str]:
    seen, budget, files, links = _Seen(limits), [0], [], []
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as e:
        raise UnsafeArchiveError("The runtime archive is not a valid ZIP file") from e
    with zf:
        for info in zf.infolist():
            for raw in (info.orig_filename, info.filename):
                if "\\" in raw:
                    raise UnsafeArchiveError(
                        f"Backslash path in runtime archive: {raw!r}"
                    )
            if info.flag_bits & 0x1:
                raise UnsafeArchiveError("Encrypted entries are not supported")
            name = _clean(info.filename)
            if not name:
                continue
            seen.add(name)
            mode = info.external_attr >> 16
            if info.is_dir():
                (dest / name).mkdir(parents=True, exist_ok=True)
            elif mode and stat.S_ISLNK(mode):
                links.append(
                    (name, _link_target(name, zf.read(info)[:4096].decode("utf-8")))
                )
            else:
                with zf.open(info) as src:
                    _write(src, dest / name, budget, limits)
                _executable_bit(dest / name, mode)
                files.append(name)
    _materialise_links(dest, links)
    return files + [n for n, _ in links]


def extract_tar_gz(
    archive: Path,
    dest: Path,
    limits: Limits = DEFAULT_LIMITS,
    *,
    compression: str = "gz",
) -> list[str]:
    """tar.gz, or tar.bz2 with ``compression="bz2"`` (speech engine and voice
    archives); the same rules either way."""
    seen, budget, files, links = _Seen(limits), [0], [], []
    try:
        tf = (
            tarfile.open(archive, mode="r:bz2")
            if compression == "bz2"
            else tarfile.open(archive, mode="r:gz")
        )
    except (tarfile.TarError, OSError, EOFError) as e:
        raise UnsafeArchiveError(
            f"The runtime archive is not a valid tar.{compression} file"
        ) from e
    with tf:
        for member in tf:
            if "\\" in member.name:
                raise UnsafeArchiveError(
                    f"Backslash path in runtime archive: {member.name!r}"
                )
            name = _clean(member.name)
            if not name:
                continue
            seen.add(name)
            if member.isdir():
                (dest / name).mkdir(parents=True, exist_ok=True)
            elif member.issym() or member.islnk():
                target = member.linkname
                links.append(
                    (
                        name,
                        _link_target(name, target)
                        if member.issym()
                        else _clean(target),
                    )
                )
            elif member.isfile():
                src = tf.extractfile(member)
                if src is None:
                    raise UnsafeArchiveError(f"Unreadable entry: {name!r}")
                with src:
                    _write(src, dest / name, budget, limits)
                _executable_bit(dest / name, member.mode)
                files.append(name)
            else:
                raise UnsafeArchiveError(f"Special file in runtime archive: {name!r}")
    _materialise_links(dest, links)
    return files + [n for n, _ in links]


def extract(
    archive: Path, fmt: str, dest: Path, limits: Limits = DEFAULT_LIMITS
) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        names = _extract(archive, fmt, dest, limits)
    except (tarfile.TarError, EOFError, zipfile.BadZipFile) as e:
        raise UnsafeArchiveError("The archive is damaged or truncated") from e
    if names is None:
        raise UnsafeArchiveError(f"Unsupported runtime archive format {fmt!r}")
    for name in names:  # belt and braces: nothing landed outside the staging dir
        if not is_within(dest / name, dest):
            raise UnsafeArchiveError(
                f"Extracted path escapes the staging folder: {name!r}"
            )
    return names


def _extract(archive: Path, fmt: str, dest: Path, limits: Limits) -> list[str] | None:
    return (
        extract_zip(archive, dest, limits)
        if fmt == "zip"
        else extract_tar_gz(archive, dest, limits)
        if fmt == "tar.gz"
        else extract_tar_gz(archive, dest, limits, compression="bz2")
        if fmt == "tar.bz2"
        else None
    )
