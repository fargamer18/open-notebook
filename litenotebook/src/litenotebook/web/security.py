"""Protections for a single-user app that listens on loopback only.

- Host validation: only loopback host names are served, which also defeats
  DNS-rebinding attacks from other websites.
- Origin validation: state-changing requests must come from this origin
  (Sec-Fetch-Site, Origin, then Referer are checked). Requests carrying none
  of these headers are non-browser local clients (curl, scripts) and are
  allowed; browsers always send at least one on cross-site POSTs.
- Strict security headers, including a CSP that allows only same-origin
  scripts and styles (no inline code, no CDNs).
"""

from __future__ import annotations

from markupsafe import escape
from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; form-action 'self'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


REJECT_PAGE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    "<title>Not allowed · LiteNotebook</title>"
    '<link rel="stylesheet" href="/static/app.css"></head><body>'
    '<main class="container" id="main"><section class="panel error-page">'
    "<h1>Not allowed</h1><p>{}</p>"
    "<p>LiteNotebook only accepts requests from its own pages on this computer.</p>"
    '<p><a class="btn" href="/">Go to notebooks</a></p>'
    "</section></main></body></html>"
)


def host_name(host_header: str) -> str:
    """Host without port; handles bracketed IPv6 literals."""
    host_header = host_header.strip().lower()
    if host_header.startswith("["):
        return host_header[1 : host_header.find("]")] if "]" in host_header else ""
    return host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header


def same_origin_problem(request: Request) -> str | None:
    """Why this unsafe request is not same-origin, or None if it is acceptable."""
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site not in ("same-origin", "none"):
        return f"cross-site request (Sec-Fetch-Site: {fetch_site})"
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
    origin = request.headers.get("origin")
    if origin is not None:
        return None if origin == expected else f"origin {origin!r} not allowed"
    referer = request.headers.get("referer")
    if referer is not None and not (
        referer == expected or referer.startswith(expected + "/")
    ):
        return "referer from another origin"
    return None


class LocalOnlyMiddleware:
    """Pure ASGI middleware (no response buffering, so streamed answers and
    client-disconnect detection pass straight through)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if host_name(headers.get("host", "")) not in ALLOWED_HOSTS:
            await self._reject("Invalid host header", 400, scope, receive, send)
            return
        if scope["method"] not in SAFE_METHODS:
            problem = same_origin_problem(Request(scope))
            if problem:
                await self._reject(f"Forbidden: {problem}", 403, scope, receive, send)
                return

        async def send_secure(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    if name.lower() not in response_headers:
                        response_headers.append(name, value)
            await send(message)

        await self.app(scope, receive, send_secure)

    @staticmethod
    async def _reject(
        text: str, status: int, scope: Scope, receive: Receive, send: Send
    ) -> None:
        accept = Headers(scope=scope).get("accept", "")
        response: Response
        if "text/html" in accept:  # a browser navigation gets a readable page
            response = HTMLResponse(
                REJECT_PAGE.format(escape(text)), status_code=status
            )
        else:
            response = PlainTextResponse(text, status_code=status)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        await response(scope, receive, send)
