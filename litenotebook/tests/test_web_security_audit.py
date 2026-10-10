"""Whole-app web security sweep (Phase 16).

Per-feature tests already cover escaping, scoping and same-origin checks for
each route family; this module sweeps *every* registered route so that a new
route cannot silently miss the protections:

* every state-changing route refuses cross-origin and foreign-host requests
  before any handler runs;
* every response (pages, JSON, static files, errors) carries the security
  headers and the strict CSP;
* unexpected errors never show a traceback, exception text or file path;
* no debug mode, no interactive docs, no OpenAPI schema;
* static files cannot escape their folder;
* the default bind address is loopback.
"""

from __future__ import annotations

import re

import httpx
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from conftest import FakeEmbedder
from litenotebook.config import load_settings
from litenotebook.web.app import create_app
from litenotebook.web.security import SECURITY_HEADERS

BASE = "http://127.0.0.1:5056"


@pytest.fixture
def app(tmp_path):  # type: ignore[no-untyped-def]
    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "data")})
    return create_app(
        settings,
        env={},
        llm_transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": []})
        ),
        embedder_factory=FakeEmbedder,
    )


@pytest.fixture
def client(app):  # type: ignore[no-untyped-def]
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as c:
        yield c


def _flat(routes):  # type: ignore[no-untyped-def]
    for route in routes:
        inner = getattr(route, "original_router", None)  # FastAPI included routers
        if inner is not None:
            yield from _flat(inner.routes)
        else:
            yield route


def _routes(app, methods: set[str]) -> list[tuple[str, str]]:  # type: ignore[no-untyped-def]
    found = []
    for route in _flat(app.routes):
        if isinstance(route, APIRoute):
            for method in sorted(route.methods & methods):
                path = re.sub(r"\{[^}]+\}", "1", route.path)
                found.append((method, path))
    return found


def test_every_state_changing_route_refuses_cross_origin(app, client):
    unsafe = _routes(app, {"POST", "PUT", "PATCH", "DELETE"})
    assert len(unsafe) >= 55  # the sweep really covers the app
    for method, path in unsafe:
        for headers in (
            {"Origin": "http://evil.example"},
            {"Origin": "null"},
            {"Origin": "http://127.0.0.1:8080"},  # another local server
            {"Sec-Fetch-Site": "cross-site"},
            {"Sec-Fetch-Site": "same-site"},
            {"Referer": "http://evil.example/page"},
        ):
            r = client.request(method, path, headers=headers)
            assert r.status_code == 403, (method, path, headers, r.status_code)


def test_every_route_refuses_foreign_host_headers(app, client):
    routes = _routes(app, {"GET", "POST"})
    for method, path in routes:
        for host in (
            "evil.example",
            "192.168.1.20:5056",
            "127.0.0.1.nip.io",
            "localhost.",
        ):
            r = client.request(method, path, headers={"Host": host})
            assert r.status_code == 400, (method, path, host)


def test_security_headers_on_pages_json_static_and_errors(client):
    for path in (
        "/",
        "/settings",
        "/healthz",
        "/livez",
        "/static/app.css",
        "/static/htmx.min.js",
        "/notebooks/999999",
        "/notebooks/abc",
        "/no/such/page",
    ):
        r = client.get(path, headers={"Accept": "text/html"})
        for name, value in SECURITY_HEADERS.items():
            assert r.headers.get(name) == value, (path, name)
    csp = SECURITY_HEADERS["Content-Security-Policy"]
    assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp


def test_unexpected_errors_show_no_traceback_or_details(app, client, monkeypatch):
    from litenotebook.web.routes import notebooks as nb_routes

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("secret detail /home/someone/private.pdf sk-abc123")

    monkeypatch.setattr(nb_routes, "_index", boom)
    for accept in ("text/html", "application/json"):
        r = client.get("/", headers={"Accept": accept})
        assert r.status_code == 500
        for leak in (
            "Traceback",
            "secret detail",
            "/home/someone",
            "sk-abc123",
            "RuntimeError",
        ):
            assert leak not in r.text, (accept, leak)


def test_no_debug_mode_docs_or_schema(app, client):
    assert app.debug is False
    for path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
        assert client.get(path).status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/static/../litenotebook.sqlite3",
        "/static/%2e%2e/%2e%2e/config.py",
        "/static/..%2f..%2fweb%2fapp.py",
        "/static/%2e%2e%5c%2e%2e%5capp.py",
        "/static//etc/passwd",
        "/static/....//....//etc/passwd",
    ],
)
def test_static_files_cannot_escape_their_folder(client, path):
    r = client.get(path)
    assert r.status_code in (400, 404), r.status_code
    assert "def create_app" not in r.text and "root:" not in r.text


def test_default_bind_is_loopback(tmp_path):
    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(tmp_path / "d")})
    assert settings.host == "127.0.0.1"


def test_path_parameters_reject_non_integers_and_huge_values(client):
    for path in (
        "/notebooks/1%20OR%201=1",
        "/notebooks/-1",
        "/notebooks/99999999999999999999999",
        "/notebooks/1/sources/..%2f..%2f/read",
        "/notebooks/1/sources/1/pages/0/render",
        "/notebooks/1/sources/1/pages/-5/render",
    ):
        r = client.get(path)
        assert r.status_code in (404, 422), (path, r.status_code)
        assert "Traceback" not in r.text
