"""Portable names: archive entry paths and download file names.

Archive entry names are always forward-slash relative paths made only of
application-generated components. These checks are written against the ZIP
spec and both Windows and POSIX rules, independent of the OS running them.
"""

from __future__ import annotations

import re
import unicodedata

_DRIVE = re.compile(r"^[A-Za-z]:")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
MAX_ENTRY_NAME = 255


def archive_name_problem(name: str) -> str | None:
    """Why ``name`` is not a safe portable archive path, or None if it is."""
    if not name:
        return "empty path"
    if len(name) > MAX_ENTRY_NAME:
        return "path too long"
    if _CONTROL.search(name):
        return "control or NUL character in path"
    if "\\" in name:
        return "backslash in path (Windows separator)"
    if name.startswith("//"):
        return "UNC path"
    if name.startswith("/"):
        return "absolute path"
    if _DRIVE.match(name):
        return "drive-letter path"
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return "relative traversal or empty component"
    for p in parts:
        if p.endswith((" ", ".")):
            return "component ends with a space or dot (not portable to Windows)"
        if p.split(".")[0].lower() in WINDOWS_RESERVED:
            return "reserved Windows device name"
        if ":" in p:
            return "colon in path (alternate data stream on Windows)"
    return None


def portable_key(name: str) -> str:
    """Case- and Unicode-insensitive identity: two entries with the same key
    would overwrite each other on Windows (and on macOS)."""
    return unicodedata.normalize("NFC", name).casefold()


def download_name(text: str, fallback: str = "notebook") -> str:
    """ASCII, filesystem-safe fragment for a Content-Disposition file name."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_text).strip("-.")[:60]
    if not slug or slug.split(".")[0].lower() in WINDOWS_RESERVED:
        return fallback
    return slug
