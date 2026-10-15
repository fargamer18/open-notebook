"""Release browser smoke: the critical pages in a real browser (Linux, Windows).

    python tools/browser_smoke.py                       # source checkout
    python tools/browser_smoke.py --app DIST/litenotebook[.exe]   # a package
    python tools/browser_smoke.py --browser PATH --json REPORT.json --keep

Seeds a TEMPORARY data folder from the test fixtures (two PDFs, chats with
citations, a correction, notes, a Reader annotation, a quiz with an attempt
and an audio-overview transcript; no LLM, no MiniLM), starts LiteNotebook on
a free loopback port and drives headless Chromium, Chrome or Edge over the
DevTools protocol (standard library only; no Node, no Selenium, no driver).

Checks: notebooks home, skip link and shortcut dialog, workspace, Reader
(render, zoom, citation highlight, annotation), chat citation → evidence →
Reader, notes, search, memory, quiz (take, score, retry mistakes, history,
evidence → highlighted page), audio transcript citation → Reader, Global and
Local graph (search, depth, inspector navigation, evidence → highlight, list
view, idle after settling, prefers-reduced-motion), no console errors and no
request outside 127.0.0.1. With --app it ends with the app's own Quit.

Needs a source checkout (the fixtures live in tests/) and a Chromium-family
browser. Never point it at real data: it always makes its own data folder.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

RESULTS: list[tuple[bool, str]] = []


def check(ok: object, label: str) -> bool:
    RESULTS.append((bool(ok), label))
    print(("PASS " if ok else "FAIL ") + label, flush=True)
    return bool(ok)


# --- a minimal DevTools client -------------------------------------------------


class _WebSocket:
    def __init__(self, url: str) -> None:
        host, rest = url[len("ws://") :].split("/", 1)
        name, port = host.rsplit(":", 1)
        self.sock = socket.create_connection((name, int(port)), timeout=60)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        if b" 101 " not in buf.split(b"\r\n")[0]:
            raise ConnectionError("DevTools refused the WebSocket upgrade")
        self.rest = buf.split(b"\r\n\r\n", 1)[1]

    def send(self, text: str) -> None:
        data = text.encode()
        head = bytearray([0x81])
        n = len(data)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(bytes(head) + body)

    def _read(self, n: int) -> bytes:
        while len(self.rest) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("DevTools connection closed")
            self.rest += chunk
        out, self.rest = self.rest[:n], self.rest[n:]
        return out

    def recv(self) -> str:
        parts = b""
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            parts += self._read(n)
            if b0 & 0x80:
                return parts.decode()


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("LITENOTEBOOK_BROWSER"):
        return os.environ["LITENOTEBOOK_BROWSER"]
    for name in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "chrome",
        "msedge",
    ):
        found = shutil.which(name)
        if found:
            return found
    for base in (
        os.environ.get("PROGRAMFILES", ""),
        os.environ.get("PROGRAMFILES(X86)", ""),
        os.environ.get("LOCALAPPDATA", ""),
    ):
        for rel in (
            r"Microsoft\Edge\Application\msedge.exe",
            r"Google\Chrome\Application\chrome.exe",
        ):
            if base and Path(base, rel).is_file():
                return str(Path(base, rel))
    return None


class Browser:
    def __init__(self, binary: str, app_origin: str) -> None:
        self.profile = tempfile.mkdtemp(prefix="ln-browser-")
        port = free_port()
        self.proc = subprocess.Popen(
            [
                binary,
                "--headless=new",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={self.profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-gpu",
                "--hide-scrollbars",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        page = None
        for _ in range(200):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json", timeout=2
                ) as r:
                    page = next(t for t in json.load(r) if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.1)
        if page is None:
            raise SystemExit("the browser did not start its DevTools endpoint")
        self.ws = _WebSocket(page["webSocketDebuggerUrl"])
        self.n = 0
        self.events: list[dict[str, Any]] = []
        self.origin = app_origin
        self.external: list[str] = []
        for domain in ("Page", "Runtime", "Log", "Network", "Performance"):
            self.call(f"{domain}.enable")
        self.call(
            "Emulation.setDeviceMetricsOverride",
            width=1366,
            height=900,
            deviceScaleFactor=1,
            mobile=False,
        )

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                result: dict[str, Any] = msg.get("result", {})
                return result
            self._event(msg)

    def _event(self, msg: dict[str, Any]) -> None:
        if msg.get("method") == "Network.requestWillBeSent":
            url = msg["params"]["request"]["url"]
            if not url.startswith((self.origin, "data:", "blob:", "about:")):
                self.external.append(url)
        self.events.append(msg)

    def js(self, expr: str) -> Any:
        r = self.call(
            "Runtime.evaluate",
            expression=expr,
            awaitPromise=True,
            returnByValue=True,
        )
        if "exceptionDetails" in r:
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:400])
        return r["result"].get("value")

    def wait(self, expr: str, timeout: float = 15.0) -> Any:
        end = time.time() + timeout
        value = None
        while time.time() < end:
            try:
                value = self.js(expr)
            except RuntimeError:
                value = None
            if value:
                return value
            time.sleep(0.1)
        return value

    def goto(self, path: str) -> None:
        url = path if path.startswith("http") else self.origin + path
        self.call("Page.navigate", url=url)
        self.wait(
            f"location.href.startsWith({json.dumps(url.split('#')[0])}) && "
            "document.readyState === 'complete'"
        )

    def click(self, selector: str) -> bool:
        return bool(
            self.js(
                f"(() => {{ const e = document.querySelector({json.dumps(selector)});"
                " if (!e) return false; e.click(); return true; })()"
            )
        )

    def key(self, key: str, code: int | None = None) -> None:
        params: dict[str, Any] = {"key": key, "code": key}
        if code:
            params["windowsVirtualKeyCode"] = code
        if len(key) == 1:
            params["text"] = key
        elif key == "Enter":
            params["text"] = "\r"  # implicit form submission needs the character
        self.call("Input.dispatchKeyEvent", type="keyDown", **params)
        params.pop("text", None)
        self.call("Input.dispatchKeyEvent", type="keyUp", **params)

    def problems(self) -> list[str]:
        self.js("1")
        out = []
        for e in self.events:
            m, p = e.get("method"), e.get("params", {})
            if m == "Runtime.exceptionThrown":
                out.append("exception: " + json.dumps(p["exceptionDetails"])[:300])
            elif m == "Runtime.consoleAPICalled" and p.get("type") in (
                "error",
                "assert",
            ):
                out.append("console: " + json.dumps(p.get("args"))[:300])
            elif m == "Log.entryAdded" and p["entry"]["level"] == "error":
                out.append("log: " + p["entry"].get("text", "")[:300])
            elif m == "Network.loadingFailed" and not p.get("canceled"):
                out.append("network: " + p.get("errorText", ""))
        self.events.clear()
        return out

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)


# --- data and server -------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def seed(data: Path, work: Path) -> dict[str, Any]:
    from graph_fixtures import build_graph_notebook
    from litenotebook.config import load_settings
    from litenotebook.db import open_db
    from litenotebook.study import audio_store
    from litenotebook.study.audio_model import Script, ScriptSection, ScriptTurn
    from litenotebook.study.model import EvidenceItem

    settings = load_settings({"LITENOTEBOOK_DATA_DIR": str(data)})
    conn = open_db(settings.db_path)
    try:
        ids: dict[str, Any] = build_graph_notebook(conn, settings, work)
        chunk = conn.execute(
            "SELECT id, page_no, text FROM chunks WHERE source_id = ? "
            "ORDER BY ord LIMIT 1",
            (ids["atlas"],),
        ).fetchone()
        oid = audio_store.create_overview(
            conn,
            ids["nb"],
            title="Zorbania overview",
            fmt="two_person",
            minutes=5,
            focus="",
            material={"sources": [{"id": ids["atlas"], "filename": "Atlas.pdf"}]},
            wpm=150,
        )
        audio_store.set_stage(conn, oid, "generating_script")
        ev = EvidenceItem(
            "S1",
            "source",
            chunk["text"],
            "Atlas.pdf",
            source_id=ids["atlas"],
            chunk_id=chunk["id"],
            page_no=chunk["page_no"],
            chunk_ord=0,
        )
        script = Script(
            "Zorbania overview",
            [
                ScriptSection(
                    "Harbour",
                    [
                        ScriptTurn("A", "Quillport is the harbour capital.", ["S1"]),
                        ScriptTurn("B", "What is it known for?", []),
                    ],
                )
            ],
            {"S1": ev},
        )
        audio_store.save_script(
            conn, oid, script, message=None, model="fixture", meta={}
        )
        ids["overview"] = oid
        ids["citation"] = conn.execute(
            "SELECT mc.id FROM message_citations mc WHERE mc.message_id = ?",
            (ids["capital_answer"],),
        ).fetchone()[0]
    finally:
        conn.close()
    return ids


def start_app(
    app: Path | None, data: Path, logs: Path, port: int
) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "LITENOTEBOOK_DATA_DIR": str(data),
        "LITENOTEBOOK_LOG_DIR": str(logs),
        "LITENOTEBOOK_PORT": str(port),
        "BROWSER": "true",
    }
    env.pop("LITENOTEBOOK_LLM_BASE_URL", None)
    if app is not None:
        cmd = [str(app), "--no-browser", "--port", str(port), "--data-dir", str(data)]
    else:
        cmd = [
            sys.executable,
            "-c",
            "import sys; from litenotebook.cli import main; sys.exit(main())",
            "serve",
            "--port",
            str(port),
        ]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def wait_up(origin: str, timeout: float = 90) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(origin + "/livez", timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except OSError:
            time.sleep(0.1)
    raise SystemExit("LiteNotebook did not start")


# --- checks ------------------------------------------------------------------------


NOTE_OPEN = (  # the note body is in the editor's textarea
    "[...document.querySelectorAll('textarea')]"
    ".some(e => e.value.includes('Quillport harbour'))"
)


def reader_highlight(b: Browser) -> Any:
    return b.wait(
        "document.querySelector('#r-img')"
        " && document.querySelector('#r-img').naturalWidth > 0"
        " && document.querySelectorAll('#r-hl .hl-cite').length > 0"
        " ? document.getElementById('r-status').textContent : null",
        20,
    )


def run_checks(b: Browser, ids: dict[str, Any]) -> dict[str, Any]:
    nb = ids["nb"]
    base = f"/notebooks/{nb}"
    metrics: dict[str, Any] = {}

    # home, skip link, shortcut dialog
    b.goto("/")
    check(
        b.wait("document.body.innerText.includes('Graph study')"),
        "notebooks home lists the notebook",
    )
    b.key("Tab", 9)
    check(
        b.js("document.activeElement && document.activeElement.className")
        == "skip-link",
        "first Tab focuses the skip link",
    )
    b.key("Enter", 13)
    check(
        b.wait("document.activeElement && document.activeElement.id === 'main'", 3),
        "skip link moves focus to the main content",
    )
    b.key("?")
    opened = b.wait("document.getElementById('shortcuts').open", 3)
    b.key("Escape", 27)
    closed = b.wait("!document.getElementById('shortcuts').open", 3)
    check(opened and closed, "'?' opens the shortcut dialog and Escape closes it")

    # workspace
    b.goto(base)
    check(
        b.wait("document.body.innerText.includes('Atlas.pdf')"),
        "workspace lists the sources",
    )

    # Reader: render, zoom, citation highlight, annotation
    b.goto(f"{base}/sources/{ids['atlas']}/read")
    rendered = b.wait(
        "document.getElementById('r-img').naturalWidth > 0"
        " && !document.getElementById('r-status').dataset.kind",
        20,
    )
    zoom0 = b.js("document.getElementById('r-zoom').textContent")
    b.click("#r-zoom-in")
    zoomed = b.wait(
        f"document.getElementById('r-zoom').textContent !== {json.dumps(zoom0)}"
        " && document.getElementById('r-img').naturalWidth > 0",
        10,
    )
    check(rendered and zoomed, f"Reader renders a page and zooms ({zoom0} → zoom in)")
    b.click("#r-fit")
    b.js("document.getElementById('r-viewport').focus()")
    b.key("PageDown", 34)
    check(
        b.wait("document.getElementById('r-page').value === '2'", 10),
        "Reader: PageDown turns the page from the keyboard",
    )
    b.goto(f"{base}/sources/{ids['atlas']}/read?citation={ids['citation']}")
    status = reader_highlight(b)
    check(
        status and "highlighted on page" in status,
        f"Reader citation link highlights the passage ({status})",
    )
    b.goto(f"{base}/sources/{ids['algebra']}/read?annotation={ids['annotation']}")
    check(
        b.wait("document.querySelectorAll('#r-hl .hl-user.is-focused').length > 0", 15),
        "Reader opens an annotation and highlights it",
    )

    # chat citation → evidence panel → Reader
    b.goto(f"{base}/chat/{ids['geo_conv']}")
    check(
        b.wait("document.body.innerText.includes('Quillport is the harbour capital')"),
        "chat shows the cited answer",
    )
    b.click("a.cite")
    href = b.wait(
        "(() => { const a = document.querySelector('#evidence a[href*=\"citation=\"]');"
        " return a ? a.getAttribute('href') : null; })()",
        10,
    )
    check(href, f"chat citation opens the evidence panel ({href})")
    if href:
        b.goto(href)
        check(reader_highlight(b), "evidence → Reader highlights the cited passage")

    # notes, search, memory
    b.goto(f"{base}/notes")
    check(
        b.wait("document.body.innerText.includes('Zorbania trip')"),
        "notes list shows the notes",
    )
    b.goto(f"{base}/notes/{ids['trip_note']}")
    check(
        b.wait(NOTE_OPEN),
        "a note opens",
    )
    b.goto(f"{base}/search?q=quillport")
    check(
        b.wait("document.body.innerText.toLowerCase().includes('quillport')"),
        "keyword search finds results",
    )
    b.goto(f"{base}/memory")
    check(
        b.wait("document.body.innerText.toLowerCase().includes('quillport')"),
        "memory list shows concepts",
    )

    # quiz: take, score, review evidence, retry mistakes, history
    b.goto(f"{base}/study/quizzes/{ids['quiz']}")
    b.click("form[action$='/start'] button")
    b.wait("location.pathname.includes('/attempts/')")
    answered = 0
    b.wait("!!document.getElementById('answer-form')")
    if b.js(
        "document.querySelector('#answer-form input[name=choice]').type === 'radio'"
    ):
        b.js("document.querySelector('#answer-form input[name=choice]').focus()")
        b.key("1")
        picked = b.js(
            "document.querySelector('#answer-form input[name=choice]:checked') !== null"
        )
        b.key("Enter", 13)
        submitted = b.wait("!!document.getElementById('feedback')", 10)
        check(picked and submitted, "quiz: '1' picks an option and Enter submits")
        answered += 1
    for _ in range(12):
        if b.js("location.pathname.endsWith('/result')"):
            break
        if b.js("!!document.getElementById('answer-form')"):
            b.click("#answer-form input[name=choice]")
            b.click("#answer-form button[type=submit]")
            b.wait(
                "!document.getElementById('answer-form')"
                " || document.readyState === 'complete'"
            )
            time.sleep(0.2)
            answered += 1
        elif b.click("a[data-next]"):
            time.sleep(0.3)
        else:
            b.click("form.quiz-finish button")
            time.sleep(0.3)
        b.wait("document.readyState === 'complete'")
    score = b.wait(
        "location.pathname.endsWith('/result') && document.querySelector('.score-n')"
        " ? document.querySelector('.score-n').textContent : null"
    )
    check(score, f"quiz taken in the browser ({answered} answers) and scored {score}")
    attempt_path = b.js("location.pathname.replace(/\\/result$/, '')")
    b.goto(attempt_path + "/review")
    pdf = b.wait(
        "(() => { const a = [...document.querySelectorAll('a')]"
        ".find(x => x.textContent.includes('Open in PDF'));"
        " return a ? a.getAttribute('href') : null; })()"
    )
    check(pdf, "quiz review links evidence to the PDF")
    if pdf:
        b.goto(pdf)
        check(reader_highlight(b), "quiz evidence → Reader highlights the passage")
    b.goto(attempt_path + "/result")
    if b.js("!!document.querySelector(\"form[action$='/retry-mistakes'] button\")"):
        b.click("form[action$='/retry-mistakes'] button")
        check(
            b.wait("document.body.innerText.includes('retrying mistakes')"),
            "Retry mistakes starts a retry attempt",
        )
    else:
        check(True, "no mistakes to retry (all answers correct)")
    b.goto(f"{base}/study/quizzes/{ids['quiz']}")
    check(
        b.wait("document.querySelectorAll('.attempt-list li').length >= 2"),
        "quiz history lists the attempts",
    )

    # audio transcript → Reader
    b.goto(f"{base}/study/audio/{ids['overview']}")
    cite = b.wait(
        "(() => { const a = document.querySelector('a.cite[href*=\"/read?\"]');"
        " return a ? a.getAttribute('href') : null; })()"
    )
    check(cite, "audio overview transcript shows source citations")
    if cite:
        b.goto(cite)
        check(
            reader_highlight(b), "transcript citation → Reader highlights the passage"
        )

    # graph: Global, search, Local depth 1/2, inspector navigation, idle
    def graph_state() -> Any:
        return b.wait(
            "window.LNGraph && LNGraph.state().nodes > 0 && LNGraph.state().alpha === 0"
            " && !LNGraph.state().pending && !LNGraph.state().quiet"
            " ? LNGraph.state() : null",
            30,
        )

    t0 = time.perf_counter()
    b.goto(f"{base}/memory/graph")
    s = graph_state()
    metrics["graph_settled_s"] = round(time.perf_counter() - t0, 2)
    check(
        s and s["mode"] == "global" and s["animatedFrames"] > 0,
        f"Global graph loads and settles ({s and s['nodes']} nodes, animated)",
    )
    frames = b.js("LNGraph.state().animatedFrames")
    task0 = {m["name"]: m["value"] for m in b.call("Performance.getMetrics")["metrics"]}
    time.sleep(2)
    task1 = {m["name"]: m["value"] for m in b.call("Performance.getMetrics")["metrics"]}
    busy = task1.get("TaskDuration", 0) - task0.get("TaskDuration", 0)
    metrics["graph_idle_task_s_per_2s"] = round(busy, 3)
    check(
        b.js("LNGraph.state().animatedFrames") == frames and busy < 0.2,
        f"settled graph is idle (no layout frames; {busy * 1000:.0f} ms of tasks"
        " in 2 s)",
    )
    b.js("document.getElementById('kg-q').focus()")
    b.call("Input.insertText", text="quill")
    hits = b.wait("document.querySelectorAll('#kg-hits button').length", 5)
    check(hits and hits >= 2, f"graph search lists {hits} matches")
    quill = f"m:{ids['quillport']}"
    b.goto(f"{base}/memory/graph?mode=local&node={quill}&depth=1")
    l1 = graph_state()
    b.goto(f"{base}/memory/graph?mode=local&node={quill}&depth=2")
    l2 = graph_state()
    check(
        l1 and l2 and l1["center"] == quill and l2["nodes"] > l1["nodes"],
        f"Local graph depth 1 → 2 ({l1 and l1['nodes']} → {l2 and l2['nodes']} nodes)",
    )

    def action(node_id: str) -> str | None:
        detail = b.js(
            f"fetch('{base}/memory/graph/node.json?id='"
            f" + encodeURIComponent({json.dumps(node_id)}))"
            ".then(r => r.json())"
        )
        acts = (detail or {}).get("actions") or []
        return acts[0]["href"] if acts else None

    note_href = action(f"m:{ids['trip_node']}")
    if note_href:
        b.goto(note_href)
    check(
        note_href and b.wait(NOTE_OPEN),
        "graph note → note editor",
    )
    chat_href = action(f"m:{ids['geo_node']}")
    if chat_href:
        b.goto(chat_href)
    check(
        chat_href and b.wait("document.body.innerText.includes('capital of Zorbania')"),
        "graph conversation → chat",
    )
    source_href = action(f"s:{ids['atlas']}")
    if source_href:
        b.goto(source_href)
    check(
        source_href
        and b.wait(
            "document.getElementById('r-img')"
            " && document.getElementById('r-img').naturalWidth > 0",
            20,
        ),
        "graph source → Reader",
    )
    view = b.js(
        f"fetch('{base}/memory/graph/view.json?types=evidence,source')"
        ".then(r => r.json())"
    )
    evidence = next((n["id"] for n in view["nodes"] if n["type"] == "evidence"), None)
    ev_href = action(evidence) if evidence else None
    if ev_href:
        b.goto(ev_href)
    check(ev_href and reader_highlight(b), "graph evidence → highlighted PDF page")

    b.goto(f"{base}/memory/graph")
    graph_state()
    b.js(
        "(() => { const d = document.getElementById('kg-list-box'); d.open = true;"
        " d.dispatchEvent(new Event('toggle')); })()"
    )
    b.wait("document.querySelectorAll('#kg-list > li').length", 5)
    b.js(
        "[...document.querySelectorAll('#kg-list > li > button')]"
        ".find(x => x.textContent.startsWith('Note: Zorbania trip')).click()"
    )
    check(
        b.wait(
            "LNGraph.state().selected && document.getElementById('kg-inspector-body')"
            ".innerText.includes('Zorbania trip')",
            5,
        ),
        "graph List view selects a node in the inspector",
    )
    b.js("document.getElementById('kg-stage').focus()")
    before = b.js("LNGraph.state().selected")
    moved = None
    for key, code in (
        ("ArrowRight", 39),
        ("ArrowLeft", 37),
        ("ArrowDown", 40),
        ("ArrowUp", 38),
    ):
        b.key(key, code)
        time.sleep(0.2)
        if b.js("LNGraph.state().selected") != before:
            moved = b.js("LNGraph.state().selected")
            break
    check(moved, f"graph arrow keys move between connected nodes ({before} → {moved})")

    # reduced motion
    b.call(
        "Emulation.setEmulatedMedia",
        features=[{"name": "prefers-reduced-motion", "value": "reduce"}],
    )
    b.goto(f"{base}/memory/graph")
    s = graph_state()
    check(
        s
        and s["reducedMotion"]
        and s["animatedFrames"] == 0
        and s["quietLayouts"] >= 1,
        "prefers-reduced-motion: the graph appears settled, without animation",
    )
    b.call(
        "Emulation.setEmulatedMedia",
        features=[{"name": "prefers-reduced-motion", "value": "no-preference"}],
    )
    return metrics


def quit_app(b: Browser, proc: subprocess.Popen[bytes]) -> None:
    b.goto("/settings/diagnostics")
    clicked = b.click("form[action='/app/quit'] button")
    try:
        proc.wait(30)
        gone = True
    except subprocess.TimeoutExpired:
        gone = False
    check(clicked and gone, "Quit LiteNotebook (Settings → Diagnostics) stops the app")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--app", type=Path, help="packaged executable (default: source)")
    ap.add_argument("--browser", help="Chromium, Chrome or Edge executable")
    ap.add_argument("--json", type=Path, help="write the results here")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder")
    args = ap.parse_args(argv)

    binary = find_browser(args.browser)
    if not binary:
        print("No Chromium, Chrome or Edge found; use --browser.", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="ln-browser-smoke-"))
    data, logs = work / "data", work / "logs"
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    proc = None
    browser = None
    metrics: dict[str, Any] = {}
    try:
        ids = seed(data, work)
        proc = start_app(args.app, data, logs, port)
        metrics["startup_s"] = round(wait_up(origin), 2)
        browser = Browser(binary, origin)
        try:
            metrics.update(run_checks(browser, ids))
        except Exception as e:  # report it and still print the summary
            check(False, f"unexpected error: {e!r}"[:300])
        problems = browser.problems()
        check(not problems, f"no console errors or failed requests {problems[:3]}")
        check(
            not browser.external,
            f"no request left 127.0.0.1 {browser.external[:3]}",
        )
        if args.app is not None:
            quit_app(browser, proc)
    finally:
        if browser is not None:
            browser.close()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept {work}")
    failed = [label for ok, label in RESULTS if not ok]
    print(
        f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed  {json.dumps(metrics)}"
    )
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "passed": len(RESULTS) - len(failed),
                    "total": len(RESULTS),
                    "failed": failed,
                    "metrics": metrics,
                    "mode": "package" if args.app else "source",
                    "browser": Path(binary).name,
                },
                indent=1,
            ),
            "utf-8",
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
