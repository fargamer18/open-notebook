"""Streaming multipart upload: request body → bounded writes → a temp file on disk.

Starlette's form parser spools uploads to the system temp directory before the
handler runs (``/tmp`` is RAM-backed tmpfs on many Linux systems) and cannot
enforce a size limit while receiving. This reader feeds the request stream
straight into python-multipart's low-level parser, writes the file part into
``<data>/tmp/uploads`` (same filesystem as the final location, so storing it is
a rename) and aborts as soon as the size limit is exceeded.

Nothing here trusts the client's filename or MIME type: the filename is kept
only as display metadata, and the content is validated as a PDF later.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from litenotebook.config import Settings
from litenotebook.platform_utils.filesystem import create_exclusive

FILE_FIELD = "file"
MAX_FIELDS = 10
MAX_FIELD_BYTES = 1024
FORM_OVERHEAD_BYTES = 64 * 1024  # multipart headers/boundaries/small fields
WRITE_BUFFER_BYTES = 1 << 20  # flush file data to disk in ≤1 MiB writes


class UploadError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ReceivedUpload:
    path: Path  # temp file inside the data directory; caller moves or deletes it
    filename: str  # as sent by the client: display metadata only, untrusted
    size: int
    fields: dict[str, str] = field(default_factory=dict)

    def discard(self) -> None:
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)


def _limit_message(max_bytes: int) -> str:
    return f"The file is larger than the {max_bytes // (1024 * 1024)} MiB upload limit"


class _Receiver:
    """python-multipart callbacks; collects small fields, writes the file part."""

    def __init__(self, max_bytes: int, kind: str) -> None:
        self.max_bytes = max_bytes
        self.kind = kind
        self.fields: dict[str, str] = {}
        self.filename: str | None = None
        self.size = 0
        self.pending: list[bytes] = []  # file bytes waiting to be written
        self.pending_bytes = 0
        self._header_field = b""
        self._header_value = b""
        self._headers: dict[bytes, bytes] = {}
        self._part_name: str | None = None
        self._part_is_file = False
        self._field_value = bytearray()
        self._seen_file = False
        self.error: UploadError | None = None

    def callbacks(self) -> dict[str, object]:
        return {
            "on_part_begin": self.on_part_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
        }

    def fail(self, message: str, status_code: int) -> None:
        if self.error is None:
            self.error = UploadError(message, status_code)

    def on_part_begin(self) -> None:
        self._headers = {}
        self._part_name, self._part_is_file = None, False
        self._field_value = bytearray()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        self._headers[self._header_field.strip().lower()] = self._header_value.strip()
        self._header_field, self._header_value = b"", b""

    def on_headers_finished(self) -> None:
        _, options = parse_options_header(
            self._headers.get(b"content-disposition", b"")
        )
        name = options.get(b"name", b"").decode("utf-8", "replace")
        self._part_name = name
        if b"filename" in options:
            if name != FILE_FIELD or self._seen_file:
                self.fail(f"Upload exactly one {self.kind} in the “file” field", 400)
                return
            self._seen_file, self._part_is_file = True, True
            self.filename = options[b"filename"].decode("utf-8", "replace")
        elif len(self.fields) >= MAX_FIELDS:
            self.fail("Too many form fields", 400)

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self.error is not None:
            return
        chunk = data[start:end]
        if self._part_is_file:
            self.size += len(chunk)
            if self.size > self.max_bytes:
                self.fail(_limit_message(self.max_bytes), 413)
                return
            self.pending.append(bytes(chunk))
            self.pending_bytes += len(chunk)
        else:
            self._field_value += chunk
            if len(self._field_value) > MAX_FIELD_BYTES:
                self.fail("A form field is too large", 400)

    def on_part_end(self) -> None:
        if not self._part_is_file and self._part_name and self.error is None:
            self.fields[self._part_name] = self._field_value.decode("utf-8", "replace")

    def take_pending(self) -> bytes:
        data = b"".join(self.pending)
        self.pending.clear()
        self.pending_bytes = 0
        return data


async def receive_upload(
    request: Request,
    settings: Settings,
    *,
    max_bytes: int | None = None,
    kind: str = "PDF",
    tmp_dir: Path | None = None,
) -> ReceivedUpload:
    """Stream a ``multipart/form-data`` upload to disk. Raises ``UploadError``.

    Defaults are the PDF upload; archive uploads pass their own limit, label
    and temp directory."""
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise UploadError(f"Upload the {kind} as multipart/form-data", 415)

    max_bytes = settings.max_upload_bytes if max_bytes is None else max_bytes
    limit = max_bytes + FORM_OVERHEAD_BYTES
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except ValueError as e:
            raise UploadError("Invalid Content-Length", 400) from e
        if declared_bytes > limit:  # reject before reading a single byte
            raise UploadError(_limit_message(max_bytes), 413)

    tmp_dir = settings.upload_tmp_dir if tmp_dir is None else tmp_dir
    await run_in_threadpool(tmp_dir.mkdir, parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}.part"
    receiver = _Receiver(max_bytes, kind)
    parser = MultipartParser(boundary, receiver.callbacks(), max_size=limit)  # type: ignore[arg-type]
    out: IO[bytes] = create_exclusive(tmp_path)  # binary mode on Windows too
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > limit:
                raise UploadError(_limit_message(max_bytes), 413)
            parser.write(chunk)
            if receiver.error is not None:
                raise receiver.error
            if receiver.pending_bytes >= WRITE_BUFFER_BYTES:
                await run_in_threadpool(out.write, receiver.take_pending())
        parser.finalize()
        if receiver.error is not None:
            raise receiver.error
        if receiver.pending_bytes:
            await run_in_threadpool(out.write, receiver.take_pending())
        await run_in_threadpool(out.close)
    except BaseException as e:
        out.close()
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        if isinstance(e, UploadError):
            raise
        if isinstance(e, ValueError):  # malformed multipart body
            raise UploadError("The upload was malformed; try again", 400) from e
        raise

    if receiver.filename is None:
        tmp_path.unlink(missing_ok=True)
        raise UploadError(f"Choose a {kind} file to upload", 400)
    if receiver.size == 0:
        tmp_path.unlink(missing_ok=True)
        raise UploadError("The uploaded file is empty", 400)
    return ReceivedUpload(tmp_path, receiver.filename, receiver.size, receiver.fields)


def clean_stale_uploads(settings: Settings) -> int:
    """Remove partial uploads left by an interrupted run (called at startup)."""
    removed = 0
    tmp_dir = settings.upload_tmp_dir
    if not tmp_dir.is_dir() or tmp_dir.is_symlink():
        return 0
    for path in tmp_dir.glob("*.part"):
        if path.is_file() and not path.is_symlink():
            with contextlib.suppress(OSError):
                path.unlink()
                removed += 1
    return removed
