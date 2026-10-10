"""Verified, resumable downloads of runtimes and models.

Bytes are untrusted until their SHA-256 matches the catalog. Lifecycle::

    disk-space preflight (download + reserve)
    → stream into <dest>.part in bounded chunks (Range-resume when possible)
    → size and SHA-256 check
    → atomic rename to <dest>

Only HTTPS URLs on the catalog's hosts are fetched; redirects may only go
to the same publishers' CDNs. A partial file is never renamed into place, so
nothing unverified can ever be selected. The transport is an injectable
``httpx.Client`` so tests use a local server instead of the internet.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from litenotebook.ai.catalog import host_allowed
from litenotebook.platform_utils.filesystem import (
    COPY_BUFFER_BYTES,
    fsync_dir,
    replace_file,
)

DEFAULT_RESERVE = 2 * 1024**3  # never fill the disk: keep 2 GiB free
TRANSIENT = (httpx.TransportError, httpx.TimeoutException)


class DownloadError(RuntimeError):
    """A download failed; the message is safe to show."""


class DownloadCancelled(DownloadError):
    pass


class RetryableDownloadError(DownloadError):
    """Transient: the transfer is retried (and resumed) a few times."""


class ChecksumMismatch(DownloadError):
    pass


class InsufficientSpace(DownloadError):
    pass


class NoConnection(DownloadError):
    pass


@dataclass
class Progress:
    done: int
    total: int | None
    rate_bps: float
    resumed_from: int = 0


@dataclass
class DownloadResult:
    path: Path
    size: int
    sha256: str
    resumed_from: int
    seconds: float


def human(n: float) -> str:
    """Binary units with IEC labels, like the web pages and llama.cpp's logs."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def _range_start(response: httpx.Response) -> int | None:
    """Start offset of a 206 response ("bytes 100-199/200")."""
    value = response.headers.get("content-range", "")
    unit, _, rest = value.partition(" ")
    start = rest.partition("-")[0]
    return int(start) if unit == "bytes" and start.isdigit() else None


def check_space(directory: Path, needed: int, reserve: int = DEFAULT_RESERVE) -> int:
    """Free bytes; raises if ``needed`` would eat into the reserve."""
    probe = directory
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if needed + reserve > free:
        raise InsufficientSpace(
            f"Not enough disk space: {human(needed)} needed plus {human(reserve)} kept "
            f"free, but only {human(free)} is available"
        )
    return free


def _hash_file(path: Path) -> tuple[hashlib._Hash, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(COPY_BUFFER_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest, size


class Downloader:
    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        require_https: bool = True,
        allowed_hosts: frozenset[str] | None = None,
        retries: int = 3,
        timeout_s: float = 30.0,
        reserve_bytes: int = DEFAULT_RESERVE,
        backoff_s: float = 1.0,
    ) -> None:
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=10.0), follow_redirects=True
        )
        self.require_https = require_https
        self.allowed_hosts = allowed_hosts  # tests: e.g. {"127.0.0.1"}
        # Checked before every request, redirects included, so the client
        # never even connects to a host outside the catalog's publishers.
        hooks = dict(self.client.event_hooks)
        hooks["request"] = [*hooks.get("request", []), self._check_request]
        self.client.event_hooks = hooks
        self.retries = retries
        self.reserve = reserve_bytes
        self.backoff = backoff_s

    def _allowed(self, url: httpx.URL | str, *, redirect: bool) -> bool:
        u = httpx.URL(str(url))
        if self.allowed_hosts is not None:
            return (u.scheme == "https" or not self.require_https) and (
                u.host in self.allowed_hosts
            )
        return host_allowed(str(u), redirect=redirect)

    def _check_request(self, request: httpx.Request) -> None:
        if not self._allowed(request.url, redirect=True):
            raise DownloadError("The download redirected to an unexpected site")

    def fetch(
        self,
        url: str,
        dest: Path,
        *,
        sha256: str | None,
        size: int | None,
        progress: Callable[[Progress], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> DownloadResult:
        if not self._allowed(url, redirect=False):
            raise DownloadError(
                "Refusing to download from a source outside the catalog"
            )
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        started = time.perf_counter()
        existing = part.stat().st_size if part.exists() else 0
        if size is not None and existing > size:
            part.unlink()
            existing = 0
        check_space(dest.parent, (size or 0) - existing, self.reserve)
        resumed_from = existing
        attempt = 0
        while True:
            try:
                self._transfer(url, part, size, progress, cancel, started, resumed_from)
                break
            except DownloadCancelled:
                raise  # .part kept: a later attempt resumes
            except (DownloadError, *TRANSIENT) as e:
                attempt += 1
                retryable = isinstance(e, (*TRANSIENT, RetryableDownloadError))
                if not retryable or attempt > self.retries:
                    if isinstance(e, httpx.ConnectError):
                        raise NoConnection(
                            "No internet connection (or the download site is "
                            "unreachable). You can retry later or use files you "
                            "already have."
                        ) from e
                    if isinstance(e, httpx.TimeoutException):
                        raise DownloadError("The download timed out; try again") from e
                    if isinstance(e, DownloadError):
                        raise
                    raise DownloadError(f"Download failed: {type(e).__name__}") from e
                if cancel is not None and cancel.wait(self.backoff * attempt):
                    raise DownloadCancelled("Download cancelled") from e
                if cancel is None:
                    time.sleep(self.backoff * attempt)
        digest, total = _hash_file(part)
        if size is not None and total != size:
            part.unlink(missing_ok=True)
            raise DownloadError(
                f"The download is incomplete ({human(total)} of {human(size)}); "
                "try again"
            )
        actual = digest.hexdigest()
        if sha256 is not None and actual != sha256:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(
                "The downloaded file does not match its published checksum and was "
                "deleted. Try again; if it keeps happening, the source has changed."
            )
        replace_file(part, dest)
        fsync_dir(dest.parent)
        return DownloadResult(
            dest, total, actual, resumed_from, round(time.perf_counter() - started, 3)
        )

    def _transfer(
        self,
        url: str,
        part: Path,
        size: int | None,
        progress: Callable[[Progress], None] | None,
        cancel: threading.Event | None,
        started: float,
        resumed_from: int,
    ) -> None:
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have == size:
            return  # complete already; verified by the caller
        headers = {"Range": f"bytes={have}-"} if have else {}
        with self.client.stream("GET", url, headers=headers) as response:
            for hop in [*response.history, response]:
                if not self._allowed(hop.url, redirect=True):
                    raise DownloadError("The download redirected to an unexpected site")
            status = response.status_code
            if status == 416 and have:  # range past the end: start over
                part.unlink(missing_ok=True)
                raise RetryableDownloadError("The server rejected the resume request")
            if status >= 500 or status == 429:
                raise RetryableDownloadError(
                    f"The download server returned HTTP {status}"
                )
            if status >= 400:
                raise DownloadError(f"The download server returned HTTP {status}")
            if status == 206 and _range_start(response) != have:
                part.unlink(missing_ok=True)  # a range we did not ask for
                raise RetryableDownloadError("The server sent the wrong part")
            append = status == 206
            if not append:  # no range support: the full file, from zero
                have = 0
            written = have
            last_report = 0.0
            with open(part, "ab") if append else open(part, "wb") as out:
                for chunk in response.iter_bytes(COPY_BUFFER_BYTES):
                    if cancel is not None and cancel.is_set():
                        raise DownloadCancelled("Download cancelled")
                    written += len(chunk)
                    if size is not None and written > size:
                        out.close()
                        part.unlink(missing_ok=True)
                        raise DownloadError("The server sent more data than expected")
                    out.write(chunk)
                    now = time.perf_counter()
                    if progress is not None and (now - last_report > 0.2):
                        elapsed = max(now - started, 1e-6)
                        progress(
                            Progress(
                                written,
                                size,
                                (written - resumed_from) / elapsed,
                                resumed_from,
                            )
                        )
                        last_report = now
            if progress is not None:
                elapsed = max(time.perf_counter() - started, 1e-6)
                progress(
                    Progress(
                        written, size, (written - resumed_from) / elapsed, resumed_from
                    )
                )
        if size is not None and written < size:
            raise RetryableDownloadError(
                "The connection closed before the download finished"
            )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.client.close()
