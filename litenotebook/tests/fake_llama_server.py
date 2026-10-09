"""A stand-in for llama-server in tests (never the real llama.cpp).

Accepts llama-server's argument names, answers ``--version`` and
``--list-devices``, serves /health (503 while "loading", then 200),
/v1/models and a streaming /v1/chat/completions with llama.cpp-style
timings. Behaviour knobs come from the environment:

FAKE_LLAMA_MODE   ok | oom_gpu (OOM whenever -ngl > 0) | oom_ctx (OOM when
                  -c > 2048) | crash | port_error | hang
FAKE_LLAMA_LOAD_S seconds of 503 before ready (default 0.3)
FAKE_LLAMA_STATE  JSON file where argv, pid and CUDA_VISIBLE_DEVICES are written
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def arg(argv: list[str], name: str, default: str) -> str:
    return argv[argv.index(name) + 1] if name in argv else default


def main() -> int:
    argv = sys.argv[1:]
    if "--version" in argv:
        print("version: 11146 (build 11146, commit fake)", file=sys.stderr)
        return 0
    if "--list-devices" in argv:
        print("Available devices:\n  CUDA0: Fake GPU (4096 MiB, 3000 MiB free)")
        return 0
    mode = os.environ.get("FAKE_LLAMA_MODE", "ok")
    port = int(arg(argv, "--port", "8080"))
    alias = arg(argv, "--alias", "fake")
    ngl = int(arg(argv, "-ngl", "0"))
    ctx = int(arg(argv, "-c", "4096"))
    state = os.environ.get("FAKE_LLAMA_STATE")
    if state:
        with open(state, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "argv": argv,
                    "pid": os.getpid(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                f,
            )
    print(f"main: loading model {arg(argv, '-m', '?')}", flush=True)
    if mode == "crash":
        print("Segmentation fault", file=sys.stderr, flush=True)
        return 139
    if mode == "port_error":
        print("couldn't bind HTTP server socket", file=sys.stderr, flush=True)
        return 1
    if (mode == "oom_gpu" and ngl > 0) or (mode == "oom_ctx" and ctx > 2048):
        print(
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2048 MiB on "
            "device 0: cudaMalloc failed: out of memory",
            file=sys.stderr,
            flush=True,
        )
        return 1
    ready_at = time.monotonic() + float(os.environ.get("FAKE_LLAMA_LOAD_S", "0.3"))
    if mode == "hang":
        ready_at = float("inf")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def _json(self, code: int, payload: object) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                if time.monotonic() < ready_at:
                    self._json(503, {"error": {"message": "Loading model"}})
                else:
                    self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"data": [{"id": alias, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            if request.get("model") not in (alias, None):
                self._json(404, {"error": {"message": "model not found"}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            words = ["<think>", "</think>", "ready"] + (
                [" and", " more"] if (request.get("max_tokens") or 0) > 50 else []
            )
            for word in words:
                event = {"choices": [{"delta": {"content": word}}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)
            final = {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "timings": {"predicted_per_second": 42.5},
                "usage": {"completion_tokens": len(words)},
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"main: server is listening on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
