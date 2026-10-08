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

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

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


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if host_name(request.headers.get("host", "")) not in ALLOWED_HOSTS:
            return self._secure(
                PlainTextResponse("Invalid host header", status_code=400)
            )
        if request.method not in SAFE_METHODS:
            problem = same_origin_problem(request)
            if problem:
                return self._secure(
                    PlainTextResponse(f"Forbidden: {problem}", status_code=403)
                )
        return self._secure(await call_next(request))

    @staticmethod
    def _secure(response: Response) -> Response:
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response
