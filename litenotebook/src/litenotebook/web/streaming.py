"""Server-sent events over a POST response, with explicit disconnect handling.

Starlette's ``StreamingResponse`` may notice a vanished client only when the
next write fails. ``EventStream`` watches ``http.disconnect`` while streaming
and cancels the generator right away, then always closes it and runs
``on_close`` (even if the generator never started), so the generation lock is
released and the message never stays "generating".
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator, Callable

import anyio
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

log = logging.getLogger(__name__)


class EventStream(Response):
    media_type = "text/event-stream"

    def __init__(
        self,
        events: AsyncGenerator[bytes, None],
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        # Not Response.__init__: it would declare "content-length: 0" for the
        # empty body, and a streamed body must not carry a Content-Length.
        self.status_code = 200
        self.background = None
        self.body = b""
        self.raw_headers = [
            (b"content-type", b"text/event-stream; charset=utf-8"),
            (b"cache-control", b"no-store"),
            (b"x-accel-buffering", b"no"),
        ]
        self.events = events
        self.on_close = on_close
        self.disconnected = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with anyio.create_task_group() as tg:

                async def watch_disconnect() -> None:
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            self.disconnected = True
                            tg.cancel_scope.cancel()
                            return

                tg.start_soon(watch_disconnect)
                await send(
                    {
                        "type": "http.response.start",
                        "status": self.status_code,
                        "headers": self.raw_headers,
                    }
                )
                async for chunk in self.events:
                    await send(
                        {"type": "http.response.body", "body": chunk, "more_body": True}
                    )
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )
                tg.cancel_scope.cancel()
        except* OSError:  # the client went away while we were writing
            self.disconnected = True
        finally:
            with anyio.CancelScope(shield=True):
                with contextlib.suppress(Exception):
                    await self.events.aclose()
                if self.on_close is not None:
                    try:
                        self.on_close()
                    except Exception:
                        log.exception("Stream cleanup failed")
