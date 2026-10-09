"""Phase 12 test fixtures: a local HTTP file server and a fake llama-server.

Nothing here touches the internet: every download test fetches from
``FileServer`` on 127.0.0.1, and runtime tests launch ``fake_llama_server.py``
(a tiny OpenAI-compatible server) instead of llama.cpp.
"""

from __future__ import annotations

import hashlib
import json
import socket
import stat
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_SERVER = HERE / "fake_llama_server.py"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Route:
    body: bytes
    ranges: bool = True  # honour Range requests
    fail_first: int = 0  # answer HTTP 500 this many times
    truncate_first: int = 0  # send half the body, then close, this many times
    delay_s: float = 0.0  # wait before the body (timeout tests)
    redirect: str | None = None
    wrong_range: bool = False  # answer 206 for a different offset
    hits: list[dict[str, str]] = field(default_factory=list)


class FileServer:
    """Threaded HTTP server for downloads; routes are added per test."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        routes = self.routes

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # quiet
                pass

            def do_GET(self) -> None:  # noqa: N802
                route = routes.get(self.path)
                if route is None:
                    self.send_error(404)
                    return
                route.hits.append(dict(self.headers))
                if route.redirect:
                    self.send_response(302)
                    self.send_header("Location", route.redirect)
                    self.end_headers()
                    return
                if route.fail_first > 0:
                    route.fail_first -= 1
                    self.send_error(500)
                    return
                body, start = route.body, 0
                header = self.headers.get("Range")
                if header and route.ranges:
                    start = int(header.split("=")[1].split("-")[0])
                    if route.wrong_range:
                        start = max(start - 10, 0) + 1
                    if start >= len(body):
                        self.send_response(416)
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header(
                        "Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}"
                    )
                else:
                    self.send_response(200)
                payload = body[start:]
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if route.delay_s:
                    time.sleep(route.delay_s)
                if route.truncate_first > 0:
                    route.truncate_first -= 1
                    self.wfile.write(payload[: len(payload) // 2])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                try:
                    for i in range(0, len(payload), 65536):
                        self.wfile.write(payload[i : i + 65536])
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def add(self, path: str, body: bytes, **options: object) -> str:
        self.routes[path] = Route(body, **options)  # type: ignore[arg-type]
        return self.url(path)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def make_fake_runtime(directory: Path, name: str = "llama-server") -> Path:
    """An executable called ``llama-server`` that runs the fake server with
    this interpreter (POSIX shebang; tests that need it are posix_only)."""
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / name
    exe.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f"sys.argv[0] = {str(FAKE_SERVER)!r}\n"
        f"runpy.run_path({str(FAKE_SERVER)!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def fake_gguf(path: Path, *, layers: int = 4, extra_bytes: int = 0) -> Path:
    """A small, valid GGUF header (v3) with llama-style metadata."""
    import struct

    def string(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    kv = [
        ("general.architecture", 8, string("qwen3")),
        ("general.name", 8, string("Tiny Test")),
        ("qwen3.block_count", 4, struct.pack("<I", layers)),
        ("qwen3.attention.head_count", 4, struct.pack("<I", 8)),
        ("qwen3.attention.head_count_kv", 4, struct.pack("<I", 2)),
        ("qwen3.embedding_length", 4, struct.pack("<I", 512)),
        ("qwen3.context_length", 4, struct.pack("<I", 4096)),
    ]
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, kind, value in kv:
        out += string(key) + struct.pack("<I", kind) + value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out + b"\0" * extra_bytes)
    return path


def launcher() -> list[str]:
    """RuntimeManager launcher: run the fake server with this interpreter."""
    return [sys.executable, str(FAKE_SERVER)]


def fake_state(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
