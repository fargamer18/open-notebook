"""Release stress and measurement runs (Phase 16). Linux; source checkout.

    python tools/release_stress.py stress   [--out REPORT.json] [--keep]
    python tools/release_stress.py offline  [--out REPORT.json]
    python tools/release_stress.py lifecycle DIST [--runtime-archive FILE]

``stress``    a large synthetic notebook (1,000+ chunks, many PDFs, notes, a
              long conversation with citations, quiz attempts, several audio
              transcripts) served by the real app; measures ingestion,
              searches, pages, Reader, backup, start-up, SQL statements and
              HTML size per page (N+1 / giant-DOM check) and the server's
              RSS across repeated work (leak check).
``offline``   the server and every page run inside a network namespace with
              only loopback (``unshare -rn``), traced with ``strace`` for any
              non-loopback ``connect()``.
``lifecycle`` the BUILT app: repeated start/quit, port collision, SIGKILL of
              the app while its embedding worker runs, SIGKILL of the worker,
              zombie and orphan checks.

Everything runs in a temporary folder with a fake OpenAI-compatible model
(no real Qwen). MiniLM, when cached, is the real embedding worker. Numbers
are observations on the machine that runs this, not pass/fail budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
PY = sys.executable
REPORT: dict[str, object] = {}


def note(key: str, value: object) -> None:
    REPORT[key] = value
    print(
        f"{key}: {json.dumps(value) if not isinstance(value, str) else value}",
        flush=True,
    )


# --- fake model ---


class FakeModel(BaseHTTPRequestHandler):
    """Chat → short cited answer; quiz / audio → the test fixtures' JSON."""

    def log_message(self, *a: object) -> None:
        pass

    def _json(self, obj: object) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/props"):
            return self._json({"default_generation_settings": {"n_ctx": 8192}})
        return self._json({"data": [{"id": "fake-qwen"}]})

    def do_POST(self) -> None:  # noqa: N802
        from audio_fixtures import outline_for, script_for
        from quiz_fixtures import default_reply

        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        schema = json.dumps((body.get("response_format") or {}).get("json_schema", {}))
        if '"questions"' in schema:
            content = json.dumps(default_reply(prompt))
        elif '"points"' in schema:
            content = json.dumps(outline_for(prompt))
        elif '"turns"' in schema:
            content = json.dumps(script_for(prompt))
        else:
            content = (
                "The capital of Zorbania is Quillport [S1]. "
                "Velt rises in the Orrin mountains [S2]."
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(content), 80):
            piece = {"choices": [{"delta": {"content": content[i : i + 80]}}]}
            self.wfile.write(f"data: {json.dumps(piece)}\n\n".encode())
        end = {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "model": "fake-qwen",
        }
        self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())


def start_fake_model() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


# --- HTTP helpers -------------------------------------------------------------


class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def req(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        files: tuple[str, str, bytes] | None = None,
        headers: dict | None = None,
        timeout: float = 600,
    ) -> tuple[int, bytes, float]:
        hdrs = {"Origin": self.base, "Accept": "text/html", **(headers or {})}
        body = None
        if files:
            import uuid

            b = uuid.uuid4().hex
            field, name, blob = files
            body = (
                (
                    f'--{b}\r\nContent-Disposition: form-data; name="{field}"; '
                    f'filename="{name}"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode()
                + blob
                + f"\r\n--{b}--\r\n".encode()
            )
            hdrs["Content-Type"] = f"multipart/form-data; boundary={b}"
        elif data is not None:
            body = urllib.parse.urlencode(data, doseq=True).encode()
            hdrs["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(
            self.base + path, data=body, method=method, headers=hdrs
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                out = r.read()
                return r.status, out, time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            return e.code, e.read(), time.perf_counter() - t0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def rss_mib(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 1)
    return 0.0


def children(pid: int) -> list[tuple[int, str, str]]:
    """(pid, state, comm) of every descendant."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = Path(f"/proc/{d}/stat").read_text()
        except OSError:
            continue
        comm = stat[stat.index("(") + 1 : stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2 :].split()
        out.append((int(d), fields[0], int(fields[1]), comm))
    tree, frontier = [], {pid}
    while frontier:
        nxt = {p for p, _, ppid, _ in out if ppid in frontier}
        tree += [(p, st, c) for p, st, ppid, c in out if ppid in frontier]
        frontier = nxt
    return tree


def wait_up(base: str, timeout: float = 90) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + "/livez", timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except OSError:
            time.sleep(0.05)
    raise SystemExit("server did not start")


def serve(
    data: Path,
    llm: str,
    port: int,
    extra: list[str] | None = None,
    env_extra: dict | None = None,
) -> subprocess.Popen:
    env = {
        **os.environ,
        "LITENOTEBOOK_DATA_DIR": str(data),
        "LITENOTEBOOK_LLM_BASE_URL": llm,
        "LITENOTEBOOK_MODE": "user",
        **(env_extra or {}),
    }
    cmd = [
        *(extra or []),
        str(Path(PY).parent / "litenotebook"),
        "serve",
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )


# --- data ---


def make_big_pdf(path: Path, pages: int) -> Path:
    import pymupdf

    from quiz_fixtures import PDF_A

    words = (
        "harbour delta glacier ledger quorum lattice meridian ember solace "
        "matrix vector pivot echelon basis kernel rank"
    ).split()
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=595, height=842)
        if i < len(PDF_A):
            text = PDF_A[i]
        else:
            paragraphs = []
            for p in range(5):
                seed = i * 7 + p
                sent = " ".join(words[(seed + k * 3) % len(words)] for k in range(18))
                paragraphs.append(
                    f"Section {i}.{p}: The {sent} record number {seed}. "
                    f"Entry {seed} describes the {words[seed % len(words)]} "
                    f"of volume {i} in detail for the archive."
                )
            text = "\n\n".join(paragraphs)
        page.insert_textbox(pymupdf.Rect(40, 40, 555, 802), text, fontsize=9)
    doc.save(path)
    doc.close()
    return path


def seed_database(data: Path, tmp: Path) -> dict[str, int]:
    """Notes, a long cited conversation and quiz attempts (direct, fast)."""
    from litenotebook.chat import CitedChunk, complete_message, start_exchange
    from litenotebook.config import load_settings
    from litenotebook.db import open_db
    from litenotebook.notes import create_note
    from litenotebook.study import store
    from quiz_fixtures import make_ready_quiz

    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(data)})
    conn = open_db(settings.db_path)
    nb = 1
    chunks = conn.execute(
        "SELECT id, source_id, page_no, ord, text FROM chunks WHERE notebook_id = 1 "
        "ORDER BY id LIMIT 400"
    ).fetchall()
    for i in range(400):
        create_note(
            conn,
            nb,
            f"Note {i}",
            f"Study note {i} about the harbour ledger entry {i * 3}. " * 4,
        )
    conv = None
    for i in range(120):
        ex = start_exchange(conn, nb, conv, f"Question {i}: what does entry {i} say?")
        conv = ex.conversation.id
        cited = [
            CitedChunk(
                label=k + 1,
                chunk_id=c[0],
                source_id=c[1],
                source_filename="big.pdf",
                page_no=c[2],
                chunk_ord=c[3],
                chunk_text=c[4][:300],
            )
            for k, c in enumerate(chunks[i % 300 : i % 300 + 5])
        ]
        complete_message(
            conn,
            ex.assistant_message_id,
            "Entry details [S1] [S2] [S3] [S4] [S5].",
            model="fake",
            citations=cited,
        )
    quizzes = [
        make_ready_quiz(conn, nb, chunks[0][1], title=f"Quiz {i}") for i in range(4)
    ]
    attempts = 0
    for qid in quizzes:
        for _ in range(15):
            aid = store.start_attempt(conn, nb, qid)
            for item in store.get_attempt(conn, nb, aid)[1].items:
                store.answer(conn, nb, aid, item.position, [1])
            store.finish_attempt(conn, nb, aid)
            attempts += 1
    conn.close()
    return {"notes": 400, "messages": 240, "citations": 600, "quiz_attempts": attempts}


# --- stress ---


def statements_per_page(data: Path, llm: str, paths: list[str]) -> dict[str, dict]:
    """In-process: SQL statements, HTML bytes and element count per page."""
    import sqlite3

    import httpx
    from fastapi.testclient import TestClient

    from litenotebook.config import load_settings
    from litenotebook.web.app import create_app

    counter = {"n": 0}
    real = sqlite3.connect

    def counting(*a, **k):  # type: ignore[no-untyped-def]
        conn = real(*a, **k)
        conn.set_trace_callback(lambda _sql: counter.__setitem__("n", counter["n"] + 1))
        return conn

    sqlite3.connect = counting  # type: ignore[assignment]
    settings = load_settings(
        {"LITENOTEBOOK_DATA_DIR": str(data), "LITENOTEBOOK_LLM_BASE_URL": llm}
    )
    out = {}
    app = create_app(settings, env={}, llm_transport=httpx.HTTPTransport())
    with TestClient(app, base_url="http://127.0.0.1:5056") as c:
        for p in paths:
            c.get(p)  # warm
            counter["n"] = 0
            r = c.get(p, headers={"Accept": "text/html"})
            html = r.text
            out[p] = {
                "status": r.status_code,
                "sql": counter["n"],
                "kib": round(len(html.encode()) / 1024, 1),
                "elements": len(re.findall(r"<[a-zA-Z]", html)),
            }
    sqlite3.connect = real  # type: ignore[assignment]
    return out


def cmd_stress(args: argparse.Namespace) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ln-stress-"))
    data = tmp / "data"
    fake, llm = start_fake_model()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    from quiz_fixtures import PDF_A  # noqa: F401  (facts the fake quiz cites)

    big = make_big_pdf(tmp / "big ledger.pdf", 520)
    note("big_pdf", {"pages": 520, "mib": round(big.stat().st_size / 2**20, 2)})
    proc = serve(data, llm, port)
    try:
        note("first_start_s", round(wait_up(base), 2))
        c = Client(base)
        c.req("POST", "/notebooks", {"name": "Stress"})
        c.req("POST", "/notebooks", {"name": "Other"})
        rss = {"idle": rss_mib(proc.pid)}
        t0 = time.perf_counter()
        st, _, _ = c.req(
            "POST", "/notebooks/1/sources", files=("file", big.name, big.read_bytes())
        )
        import sqlite3

        def ready(nb: int = 1) -> bool:
            db = sqlite3.connect(
                f"file:{data / 'litenotebook.sqlite3'}?mode=ro", uri=True
            )
            try:
                rows = db.execute(
                    "SELECT status FROM sources WHERE notebook_id = ?", (nb,)
                ).fetchall()
            finally:
                db.close()
            return bool(rows) and all(r[0] in ("ready", "failed") for r in rows)

        while not ready():
            time.sleep(0.5)
        ingest_s = time.perf_counter() - t0
        db = sqlite3.connect(f"file:{data / 'litenotebook.sqlite3'}?mode=ro", uri=True)
        chunks, status = db.execute(
            "SELECT count(*), (SELECT status FROM sources WHERE id = 1) FROM chunks"
        ).fetchone()
        db.close()
        note(
            "ingest_big_pdf",
            {
                "status": status,
                "chunks": chunks,
                "seconds": round(ingest_s, 1),
                "chunks_per_s": round(chunks / ingest_s, 1),
            },
        )
        rss["after_ingest"] = rss_mib(proc.pid)
        from audio_fixtures import PDF_TRAP
        from conftest import make_pdf

        small = [
            make_pdf(
                tmp / f"small{i}.pdf",
                PDF_TRAP
                if i == 0
                else [f"Small document {i} about the Orrin glacier."] * 3,
            )
            for i in range(4)
        ]
        for p in small:
            c.req(
                "POST", "/notebooks/1/sources", files=("file", p.name, p.read_bytes())
            )
        while not ready():
            time.sleep(0.3)
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
        note("seeded", seed_database(data, tmp))
        proc = serve(data, llm, port)
        note("start_with_large_notebook_s", round(wait_up(base), 2))
        rss["restart_idle"] = rss_mib(proc.pid)
        # audio transcripts through the real generation path (fake model)
        db = sqlite3.connect(f"file:{data / 'litenotebook.sqlite3'}?mode=ro", uri=True)
        trap_id = db.execute(
            "SELECT id FROM sources WHERE original_filename = 'small0.pdf'"
        ).fetchone()[0]
        db.close()
        made = 0
        for _ in range(4):
            st, body, _ = c.req(
                "POST",
                "/notebooks/1/study/audio",
                {"sources": str(trap_id), "format": "two_person", "minutes": "5"},
            )
            made += st == 200
            time.sleep(0.5)
        time.sleep(3)
        note("audio_transcripts_requested", made)

        def timed(path: str, n: int = 15) -> dict[str, float]:
            times = []
            for _ in range(n):
                st, _, t = c.req("GET", path)
                assert st == 200, (path, st)
                times.append(t * 1000)
            return {
                "median_ms": round(statistics.median(times), 1),
                "p95_ms": round(sorted(times)[int(0.95 * (n - 1))], 1),
            }

        pages = {
            "notebooks": "/",
            "workspace": "/notebooks/1",
            "sources": "/notebooks/1?tab=sources",
            "chat (120 turns)": "/notebooks/1/chat",
            "notes list (400)": "/notebooks/1/notes",
            "study (4 quizzes, 60 attempts)": "/notebooks/1/study",
            "memory": "/notebooks/1/memory",
            "memory graph page": "/notebooks/1/memory/graph",
            "fast search": "/notebooks/1/search?q=harbour+ledger",
            "reader page": "/notebooks/1/sources/1/read?page=150",
            "settings": "/settings",
            "settings/ai": "/settings/ai",
        }
        db = sqlite3.connect(f"file:{data / 'litenotebook.sqlite3'}?mode=ro", uri=True)
        node = db.execute(
            "SELECT id FROM memory_nodes WHERE notebook_id = 1 ORDER BY id LIMIT 1"
        ).fetchone()
        db.close()
        if node:
            pages["memory graph.json (depth 2)"] = (
                f"/notebooks/1/memory/graph.json?node={node[0]}&depth=2"
            )
        note("page_times", {k: timed(v) for k, v in pages.items()})
        # Reader render: cold, then cached
        cold, warm = [], []
        for page in (10, 50, 200):
            cold.append(
                c.req("GET", f"/notebooks/1/sources/1/pages/{page}/render")[2] * 1000
            )
            warm.append(
                c.req("GET", f"/notebooks/1/sources/1/pages/{page}/render")[2] * 1000
            )
        note(
            "reader_render_ms",
            {
                "cold_median": round(statistics.median(cold), 1),
                "cached_median": round(statistics.median(warm), 1),
            },
        )
        st, _, t_h = c.req("GET", "/notebooks/1/search?q=harbour+ledger&engine=hybrid")
        note("hybrid_search_first_ms", round(t_h * 1000))
        hy = [
            c.req("GET", f"/notebooks/1/search?q=glacier+{i}&engine=hybrid")[2] * 1000
            for i in range(10)
        ]
        note("hybrid_search_ms", {"median": round(statistics.median(hy), 1)})
        st, body, t_b = c.req("POST", "/backup/create", {})
        note(
            "full_backup",
            {
                "status": st,
                "seconds": round(t_b, 2),
                "mib": round(len(body) / 2**20, 1),
            },
        )
        rss["after_pages"] = rss_mib(proc.pid)
        # memory: repeated work rounds
        series = []
        for rnd in range(6):
            for page in range(1, 40):
                c.req("GET", f"/notebooks/1/sources/1/pages/{page}/render?scale=1.25")
                c.req("GET", f"/notebooks/1/sources/1/pages/{page}/text")
            for i in range(20):
                c.req("GET", f"/notebooks/1/search?q=entry+{rnd * 20 + i}")
            for i in range(3):
                c.req(
                    "POST",
                    "/notebooks/1/chat/send",
                    {"question": f"Round {rnd} question {i}?", "conversation_id": ""},
                )
            c.req(
                "POST",
                "/notebooks/1/study/quizzes",
                {
                    "sources": "1",
                    "types": ["single", "multiple", "true_false"],
                    "count": "5",
                    "difficulty": "medium",
                },
            )
            time.sleep(1.5)
            st, archive, _ = c.req("POST", "/notebooks/2/export", {})
            c.req("POST", "/backup/import", files=("file", "x.zip", archive))
            series.append(rss_mib(proc.pid))
        note("rss_series_mib", series)
        growth = [b - a for a, b in zip(series, series[1:], strict=False)]
        note(
            "rss_leak_check",
            {
                "first": series[0],
                "last": series[-1],
                "max_round_growth": max(growth),
                "monotonic": all(g > 0.5 for g in growth),
            },
        )
        rss["end"] = rss_mib(proc.pid)
        note("server_rss_mib", rss)
        tree = children(proc.pid)
        note("children_at_end", [(p, s, n) for p, s, n in tree])
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
        note(
            "statements_per_page", statements_per_page(data, llm, list(pages.values()))
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
        fake.shutdown()
        if args.out:
            Path(args.out).write_text(json.dumps(REPORT, indent=1), encoding="utf-8")
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


# --- offline ---


def cmd_offline(args: argparse.Namespace) -> int:
    if not args.inner:
        if not shutil.which("unshare") or not shutil.which("strace"):
            raise SystemExit("needs unshare and strace")
        out = args.out or str(Path(tempfile.gettempdir()) / "ln-offline.json")
        cmd = [
            "unshare",
            "-rn",
            "sh",
            "-c",
            f"ip link set lo up 2>/dev/null; exec '{PY}' '{__file__}' offline --inner "
            f"--out '{out}'",
        ]
        code = subprocess.call(cmd)
        print(Path(out).read_text())
        return code
    tmp = Path(tempfile.mkdtemp(prefix="ln-offline-"))
    data = tmp / "data"
    trace = tmp / "connect.trace"
    fake, llm = start_fake_model()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = serve(
        data,
        llm,
        port,
        extra=["strace", "-f", "-qq", "-e", "trace=connect", "-o", str(trace)],
    )
    ok = []
    try:
        wait_up(base)
        c = Client(base)
        from conftest import make_pdf

        c.req("POST", "/notebooks", {"name": "Offline"})
        pdf = make_pdf(tmp / "o.pdf", ["The Velt river rises in the Orrin mountains."])
        c.req("POST", "/notebooks/1/sources", files=("file", "o.pdf", pdf.read_bytes()))
        time.sleep(25)  # ingestion with the local MiniLM worker (if installed)
        c.req("POST", "/notebooks/1/notes", {"title": "N", "content": "violet trams"})
        for path in (
            "/",
            "/setup",
            "/settings",
            "/settings/ai",
            "/settings/diagnostics",
            "/notebooks/1",
            "/notebooks/1/notes",
            "/notebooks/1/study",
            "/notebooks/1/search?q=violet",
            "/notebooks/1/search?q=Velt&engine=hybrid",
            "/notebooks/1/sources/1/read",
            "/notebooks/1/sources/1/pages/1/render",
            "/backup",
        ):
            st, _, _ = c.req("GET", path)
            ok.append((path, st))
        st, _, _ = c.req(
            "POST",
            "/notebooks/1/chat/send",
            {"question": "Where does the Velt rise?", "conversation_id": ""},
        )
        ok.append(("chat (local fake model)", st))
        st, body, _ = c.req("POST", "/notebooks/1/export", {})
        ok.append(("export", st))
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        fake.shutdown()
    import sqlite3

    db = sqlite3.connect(f"file:{data / 'litenotebook.sqlite3'}?mode=ro", uri=True)
    REPORT["source_status"] = db.execute(
        "SELECT status, coalesce(error, '') FROM sources WHERE id = 1"
    ).fetchone()
    db.close()
    lines = trace.read_text(errors="replace").splitlines() if trace.exists() else []
    inet = [ln for ln in lines if "AF_INET" in ln]
    external = [
        ln
        for ln in inet
        if not re.search(r'inet_addr\("127\.|inet_pton\(AF_INET6, "::1"', ln)
    ]
    REPORT.update(
        {
            "pages": ok,
            "inet_connects": len(inet),
            "non_loopback_connects": external[:20],
            "unix_socket_connects": len([ln for ln in lines if "AF_UNIX" in ln]),
        }
    )
    Path(args.out).write_text(json.dumps(REPORT, indent=1), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if not external and all(s == 200 for _, s in ok) else 1


# --- lifecycle (built package) ---


def cmd_lifecycle(args: argparse.Namespace) -> int:
    dist = Path(args.dist).resolve()
    folder = next(
        f
        for f in sorted(dist.glob("LiteNotebook-*-*"))
        if (f / "litenotebook-build.json").is_file()
    )
    exe = folder / "litenotebook"
    tmp = Path(tempfile.mkdtemp(prefix="ln-life-"))
    data = tmp / "data"
    port = free_port()
    env = {
        **os.environ,
        "LITENOTEBOOK_DATA_DIR": str(data),
        "LITENOTEBOOK_LOG_DIR": str(tmp / "logs"),
        "LITENOTEBOOK_PORT": str(port),
        "LITENOTEBOOK_EMBED_WORKER_IDLE_SECONDS": "600",
    }
    env.pop("LITENOTEBOOK_EMBED_WORKER", None)

    def launch() -> subprocess.Popen:
        return subprocess.Popen(
            [str(exe), "--no-browser"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def url_of(p: subprocess.Popen, timeout: float = 90) -> str:
        """The running app's URL from its instance lock (the port can differ
        from the configured one, e.g. while the old one is in TIME_WAIT)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                info = json.loads((data / ".instance.lock").read_text("utf-8"))
                if info.get("pid") == p.pid and info.get("url"):
                    wait_up(info["url"], 30)
                    return str(info["url"])
            except (OSError, ValueError):
                pass
            time.sleep(0.1)
        raise SystemExit("app did not start")

    def ours() -> list[int]:
        found = []
        roots = (str(folder), str(data))
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    if os.readlink(f"/proc/{d}/exe").startswith(roots):
                        found.append(int(d))
                except OSError:
                    pass
        return found

    results: dict[str, object] = {}
    try:
        times = []
        for _ in range(8):
            t0 = time.perf_counter()
            p = launch()
            url = url_of(p)
            times.append(round(time.perf_counter() - t0, 2))
            Client(url).req("POST", "/app/quit", {})
            p.wait(timeout=30)
        time.sleep(1)
        results["repeated_start_quit"] = {"runs": 8, "ready_s": times, "left": ours()}
        # port collision: something else listens on the configured port
        blocker = socket.socket()
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # TIME_WAIT
        blocker.bind(("127.0.0.1", port))
        blocker.listen()
        p = launch()
        time.sleep(4)
        log = (tmp / "logs" / "litenotebook.log").read_text(errors="replace")
        moved = re.findall(r"Port \d+ is in use by another program; using (\d+)", log)
        if moved:
            alt = f"http://127.0.0.1:{moved[-1]}"
            wait_up(alt)
            Client(alt).req("POST", "/app/quit", {})
        p.wait(timeout=30)
        blocker.close()
        results["port_collision"] = {
            "moved_to": moved[-1:] or None,
            "exit": p.returncode,
        }
        if args.runtime_archive:
            p = launch()
            c = Client(url_of(p))
            st, body, _ = c.req(
                "POST",
                "/settings/ai/install-embedding-runtime",
                {"path": str(Path(args.runtime_archive).resolve())},
            )
            job = re.search(rb"/ai/jobs/([0-9a-f]+)", body)
            deadline = time.time() + 300
            while job and time.time() < deadline:
                request = urllib.request.Request(
                    f"{c.base}/ai/jobs/{job.group(1).decode()}",
                    headers={"Accept": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=30) as r:
                    state = json.loads(r.read()).get("state")
                if state != "running":
                    results["runtime_install"] = state
                    break
                time.sleep(0.5)
            c.req("POST", "/notebooks", {"name": "Life"})
            from conftest import make_pdf

            pdf = make_pdf(
                tmp / "l.pdf", ["Velt river rises in the Orrin mountains."] * 3
            )
            c.req(
                "POST",
                "/notebooks/1/sources",
                files=("file", "l.pdf", pdf.read_bytes()),
            )

            def worker() -> list[int]:
                found = []
                for pid, _, _ in children(p.pid):
                    try:
                        exe_path = os.readlink(f"/proc/{pid}/exe")
                    except OSError:
                        continue
                    if "litenotebook-embed-worker" in exe_path:
                        found.append(pid)
                return found

            deadline = time.time() + 120
            while not worker() and time.time() < deadline:
                time.sleep(0.5)
            w = worker()
            results["worker_started"] = bool(w)
            if w:
                os.kill(w[0], signal.SIGKILL)  # worker crash
                time.sleep(2)
                zombies = [x for x in children(p.pid) if x[1] == "Z"]
                pdf2 = make_pdf(
                    tmp / "l2.pdf", ["Second document about the delta."] * 3
                )
                c.req(
                    "POST",
                    "/notebooks/1/sources",
                    files=("file", "l2.pdf", pdf2.read_bytes()),
                )
                deadline = time.time() + 120
                while not worker() and time.time() < deadline:
                    time.sleep(0.5)
                results["worker_crash"] = {
                    "zombies_after_kill": zombies,
                    "respawned": bool(worker()),
                }
                # SIGKILL the app itself while the worker runs: guard must reap it
                left_worker = worker()
                os.kill(p.pid, signal.SIGKILL)
                p.wait(timeout=10)
                deadline = time.time() + 20
                while time.time() < deadline and any(
                    Path(f"/proc/{x}").exists() for x in left_worker
                ):
                    time.sleep(0.5)
                results["app_sigkill"] = {
                    "worker_survivors": [
                        x for x in left_worker if Path(f"/proc/{x}").exists()
                    ],
                    "left": ours(),
                }
            else:
                c.req("POST", "/app/quit", {})
                p.wait(timeout=30)
        time.sleep(1)
        results["final_leftovers"] = ours()
    finally:
        for pid in ours():
            try:
                os.kill(pid, signal.SIGKILL)  # only processes of this temp install
            except OSError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(results, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stress")
    s.add_argument("--out")
    s.add_argument("--keep", action="store_true")
    o = sub.add_parser("offline")
    o.add_argument("--out")
    o.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    life = sub.add_parser("lifecycle")
    life.add_argument("dist")
    life.add_argument("--runtime-archive")
    life.add_argument("--out")
    args = parser.parse_args()
    return {"stress": cmd_stress, "offline": cmd_offline, "lifecycle": cmd_lifecycle}[
        args.cmd
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
