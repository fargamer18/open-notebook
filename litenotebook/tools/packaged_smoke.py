"""Smoke-test a BUILT LiteNotebook package (Linux or Windows; stdlib only).

    python tools/packaged_smoke.py dist/litenotebook [--runtime-archive FILE]
                                   [--keep] [--json OUT] [--export-to DIR]
    python tools/packaged_smoke.py --app-folder "C:\\...\\Programs\\LiteNotebook"

Runs the packaged executable — not the source tree — against a temporary data
folder whose path contains spaces and non-ASCII characters, with the browser
replaced by a recorder (the BROWSER variable), and checks:

* launch → the browser is asked to open the app's loopback URL
* /livez version = the build manifest's; pages, templates, static files load
* a second launch opens the same URL and does not start a second server
* notebook, note and fast search; a PDF upload is indexed when the
  semantic search runtime (+ MiniLM in the Hugging Face cache) is present,
  otherwise it fails with the "runtime is not installed" message
* nothing listens beyond loopback (the machine's LAN address is refused)
* archives of formats 1, 2 and 3 (created on Linux) import; the imported
  quiz and audio transcript open and the imported WAV plays (Range request)
* notebook export and import (backup); ``--export-to`` keeps the export for
  importing on another platform
* deleting the uploaded source removes its file (no open-file errors)
* with the runtime: the worker stops after its idle timeout
* Settings → Local AI and Setup start no download
* Quit LiteNotebook stops the process; no LiteNotebook process is left;
  a restart finds the same data

It never touches the user's own data folder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
FAILED: list[str] = []
RESULTS: dict[str, object] = {}


def check(ok: bool, label: str) -> None:
    print(("PASS " if ok else "FAIL ") + label, flush=True)
    if not ok:
        FAILED.append(label)


def find_app(dist: Path) -> tuple[Path, Path, dict]:
    if (dist / "litenotebook-build.json").is_file():  # an installed app folder
        folder = dist.resolve()
        exe = folder / ("LiteNotebook.exe" if IS_WINDOWS else "litenotebook")
        manifest = json.loads((folder / "litenotebook-build.json").read_text("utf-8"))
        return folder, exe, manifest
    for folder in sorted(dist.resolve().glob("LiteNotebook-*-*")):
        if folder.is_dir() and (folder / "litenotebook-build.json").is_file():
            exe = folder / ("LiteNotebook.exe" if IS_WINDOWS else "litenotebook")
            manifest = json.loads(
                (folder / "litenotebook-build.json").read_text("utf-8")
            )
            return folder, exe, manifest
    raise SystemExit(f"No built LiteNotebook app folder in {dist}")


def tiny_pdf(text: str) -> bytes:
    """A valid one-page PDF with a text layer (no third-party library)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    out += (trailer + f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


class Client:
    def __init__(self, base: str) -> None:
        self.base = base

    def req(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        body: bytes | None = None,
        ctype: str | None = None,
        follow: bool = True,
    ) -> tuple[int, dict, bytes]:
        headers = {"Origin": self.base, "Accept": "text/html"}
        if data is not None:
            body = urllib.parse.urlencode(data, doseq=True).encode()
            ctype = "application/x-www-form-urlencoded"
        if ctype:
            headers["Content-Type"] = ctype
        request = urllib.request.Request(
            self.base + path, data=body, method=method, headers=headers
        )
        opener = urllib.request.build_opener(*([] if follow else [_NoRedirect()]))
        try:
            with opener.open(request, timeout=120) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def upload(
        self, path: str, field: str, filename: str, content: bytes, mime: str
    ) -> tuple[int, dict, bytes]:
        boundary = uuid.uuid4().hex
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
                f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'
            ).encode()
            + content
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return self.req(
            "POST", path, body=body, ctype=f"multipart/form-data; boundary={boundary}"
        )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None


def wait_for(predicate, timeout: float, step: float = 0.1):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    return None


def _windows_processes() -> list[tuple[int, str]]:
    script = (
        "Get-CimInstance Win32_Process | ForEach-Object "
        '{ "$($_.ProcessId)`t$($_.ExecutablePath)" }'
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    ).stdout
    rows = []
    for line in out.splitlines():
        pid, _, exe = line.partition("\t")
        if pid.strip().isdigit() and exe.strip():
            rows.append((int(pid), exe.strip()))
    return rows


def app_processes(*folders: Path, name: str = "") -> list[int]:
    """Processes running from these folders (the app, an embedding runtime
    installed in the data folder), or whose executable contains ``name``."""
    roots = tuple(str(f.resolve()).lower() for f in folders)

    def match(exe: str) -> bool:
        e = exe.lower()
        return bool((roots and e.startswith(roots)) or (name and name in e))

    if IS_WINDOWS:
        return [pid for pid, exe in _windows_processes() if match(exe)]
    if not Path("/proc").is_dir():
        return []
    found = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                exe = os.readlink(f"/proc/{d}/exe")
            except OSError:
                continue
            if match(exe):
                found.append(int(d))
    return found


def lan_address() -> str | None:
    """This machine's non-loopback address (no packet is sent)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1: routing lookup only
            addr = s.getsockname()[0]
        except OSError:
            return None
    return None if addr.startswith("127.") else addr


def refused(host: str, port: int) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=2):
            return False
    except OSError:
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path, nargs="?")
    parser.add_argument(
        "--app-folder", type=Path, help="test an installed app folder instead"
    )
    parser.add_argument(
        "--export-to", type=Path, help="keep the exported notebook archive here"
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures",
        help="archives created on Linux to import (formats 1-3)",
    )
    parser.add_argument(
        "--runtime-archive",
        type=Path,
        help="install this embedding runtime archive first",
    )
    parser.add_argument("--keep", action="store_true", help="keep the temp folder")
    parser.add_argument("--json", type=Path, help="write timings/results here")
    args = parser.parse_args()
    target = args.app_folder or args.dist
    if target is None:
        parser.error("give the dist folder or --app-folder")
    folder, exe, manifest = find_app(target)
    print(f"Testing {folder.name} ({manifest.get('version')})")
    tmp = Path(tempfile.mkdtemp(prefix="ln-packaged-"))
    home = tmp / "Normal User José"  # spaces + non-ASCII in every path
    data, logs = home / "data", home / "logs"
    record = tmp / "browser-urls.txt"
    recorder = tmp / "record_browser.py"
    recorder.write_text(
        "import sys\n"
        "open(sys.argv[1], 'a', encoding='utf-8').write(sys.argv[2] + '\\n')\n",
        encoding="utf-8",
    )
    browser_cmd = f'"{sys.executable}" "{recorder}" "{record}" %s'
    env = dict(
        os.environ,
        LITENOTEBOOK_DATA_DIR=str(data),
        LITENOTEBOOK_LOG_DIR=str(logs),
        BROWSER=browser_cmd,
        LITENOTEBOOK_PORT="5091",
        LITENOTEBOOK_EMBED_WORKER_IDLE_SECONDS="6",
    )
    for key in ("LITENOTEBOOK_MODE", "LITENOTEBOOK_EMBED_WORKER"):
        env.pop(key, None)

    def urls() -> list[str]:
        try:
            return [u for u in record.read_text("utf-8").split("\n") if u]
        except OSError:
            return []

    # PDF engine: PDFium (with its licence texts); PyMuPDF must not be bundled
    names = [f.name.lower() for f in folder.rglob("*")]
    mupdf = sorted({n for n in names if "mupdf" in n or n in ("fitz", "fitz.py")})
    check(not mupdf, f"no PyMuPDF/MuPDF files in the package {mupdf[:5]}")
    check(
        any(n.startswith(("libpdfium", "pdfium.dll")) for n in names),
        "PDFium library bundled",
    )
    check(
        (folder / "licenses" / "pypdfium2").is_dir(),
        "PDFium licence texts shipped (licenses/pypdfium2)",
    )

    started = time.perf_counter()
    proc = subprocess.Popen(
        [str(exe)],
        env=env,
        cwd=str(tmp),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        opened = wait_for(urls, 90)
        RESULTS["launch_to_browser_s"] = round(time.perf_counter() - started, 2)
        check(bool(opened), f"browser asked to open the app ({opened})")
        if not opened:
            return 1
        base = opened[0]
        check(base.startswith("http://127.0.0.1:"), "loopback URL only")
        port = int(base.rsplit(":", 1)[1])
        lan = lan_address()
        if lan:
            check(refused(lan, port), f"not reachable on the LAN address {lan}")
        else:
            print("SKIP no LAN address to test against")
        c = Client(base)
        st, _, body = c.req("GET", "/livez")
        live = json.loads(body)
        check(
            st == 200 and live.get("version") == manifest["version"],
            f"livez reports {live.get('version')}",
        )
        st, _, body = c.req("GET", "/healthz")
        check(st in (200, 503) and b'"database"' in body, "healthz answers")
        for path in (
            "/",
            "/setup",
            "/settings",
            "/settings/ai",
            "/backup",
            "/static/app.css",
            "/static/htmx.min.js",
            "/static/icon.svg",
        ):
            st, _, _ = c.req("GET", path)
            check(st == 200, f"GET {path} → {st}")
        st, _, body = c.req("GET", "/settings/diagnostics")
        check(
            b"packaged" in body and b"Quit LiteNotebook" in body,
            "diagnostics: packaged mode, Quit offered",
        )
        expected_os = b"windows" if IS_WINDOWS else b"linux"
        check(expected_os in body.lower(), f"diagnostics names {expected_os.decode()}")
        models = data / "models"
        check(
            not models.exists() or not any(models.rglob("*.gguf")),
            "Local AI / Setup pages started no model download",
        )
        check(
            data.is_dir() and (data / "litenotebook.sqlite3").is_file(),
            "data in the chosen data folder",
        )
        check((logs / "litenotebook.log").is_file(), "log file written")
        # second launch: opens the running instance, starts no second server
        t2 = time.perf_counter()
        second = subprocess.run(
            [str(exe)],
            env=env,
            cwd=str(tmp),
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        RESULTS["second_launch_s"] = round(time.perf_counter() - t2, 2)
        check(
            second.returncode == 0 and urls()[-1] == base and len(urls()) == 2,
            "second launch opens the same page and exits",
        )
        check(proc.poll() is None, "first instance still running")
        if args.runtime_archive:
            st, _, body = c.req(
                "POST",
                "/settings/ai/install-embedding-runtime",
                {"path": str(args.runtime_archive.resolve())},
            )
            m = re.search(rb"/ai/jobs/([0-9a-f]+)", body) or re.search(
                rb"job=([0-9a-f]+)", body
            )
            ok = False
            if m:
                job_id = m.group(1).decode()
                job = wait_for(lambda: _json_job(c, job_id), 600, 0.5)
                ok = bool(job and job.get("state") == "done")
            check(ok, "embedding runtime installed from its archive")
        # notebook, note, fast search
        st, h, _ = c.req("POST", "/notebooks", {"name": "Packaged smoke"}, follow=False)
        st, _, body = c.req("GET", "/")
        nb = int(re.search(rb'href="/notebooks/(\d+)', body).group(1))
        st, h, _ = c.req(
            "POST",
            f"/notebooks/{nb}/notes",
            {"title": "Harbour", "content": "Quillport trams are violet."},
            follow=False,
        )
        check(st == 303, "note created")
        st, _, body = c.req("GET", f"/notebooks/{nb}/search?q=violet")
        check(st == 200 and b"Harbour" in body, "fast search finds the note")
        # PDF upload: indexed with the runtime, a clear message without it
        st, _, _ = c.upload(
            f"/notebooks/{nb}/sources",
            "file",
            "Zorbania atlas.pdf",
            tiny_pdf("The capital of Zorbania is Quillport."),
            "application/pdf",
        )
        check(st in (200, 303), f"PDF upload accepted ({st})")

        def source_state() -> tuple[str, str] | None:
            import sqlite3

            db = sqlite3.connect(
                f"file:{(data / 'litenotebook.sqlite3').as_posix()}?mode=ro", uri=True
            )
            try:
                row = db.execute(
                    "SELECT status, coalesce(error, '') FROM sources "
                    "WHERE notebook_id = ?",
                    (nb,),
                ).fetchone()
            finally:
                db.close()
            return (row[0], row[1]) if row and row[0] in ("ready", "failed") else None

        state = wait_for(source_state, 300, 0.5)
        RESULTS["pdf_state"] = state[0] if state else None
        if state and state[0] == "ready":
            check(True, "PDF indexed (semantic search runtime and MiniLM present)")
            st, _, body = c.req("GET", f"/notebooks/{nb}/search?q=Quillport")
            check(b"Zorbania" in body, "search finds the PDF passage")
            sid = re.search(
                rb"/sources/(\d+)/read", c.req("GET", f"/notebooks/{nb}?tab=sources")[2]
            )
            if sid:
                st, _, _ = c.req(
                    "GET", f"/notebooks/{nb}/sources/{sid.group(1).decode()}/read"
                )
                check(st == 200, "Reader opens the PDF")
                st, _, png = c.req(
                    "GET",
                    f"/notebooks/{nb}/sources/{sid.group(1).decode()}/pages/1/render",
                )
                check(st == 200 and png[:4] == b"\x89PNG", "Reader renders page 1")
                st, _, text = c.req(
                    "GET",
                    f"/notebooks/{nb}/sources/{sid.group(1).decode()}/pages/1/text",
                )
                check(
                    st == 200 and b'"t":"Zorbania"' in text.replace(b" ", b""),
                    "Reader text layer has word boxes (PDFium geometry)",
                )
        else:
            message = state[1] if state else ""
            RESULTS["pdf_error"] = message
            check(
                bool(state)
                and (
                    "runtime is not installed" in message
                    or "not available locally" in message
                ),
                "without the runtime/model the PDF fails with a clear message "
                f"({message[:80]})",
            )
        if state and state[0] == "ready" and args.runtime_archive:
            gone = wait_for(
                lambda: not app_processes(name="litenotebook-embed-worker"), 60, 0.5
            )
            check(bool(gone), "embedding worker stopped after its idle timeout")
        # backup: export and import
        st, h, archive = c.req("POST", f"/notebooks/{nb}/export", {})
        check(
            st == 200 and archive[:2] == b"PK",
            f"notebook exported ({len(archive)} bytes)",
        )
        st, _, body = c.upload(
            "/backup/import", "file", "export.zip", archive, "application/zip"
        )
        check(st == 200 and b"Packaged smoke" in body, "notebook imported again")
        if args.export_to:
            args.export_to.mkdir(parents=True, exist_ok=True)
            out = args.export_to / f"export-from-{sys.platform}.zip"
            out.write_bytes(archive)
            print(f"INFO export kept for cross-platform import: {out}")
        # archives created on another platform (the committed Linux fixtures)
        for fixture, label in (
            ("phase12_notebook_v1.zip", "format 1"),
            ("phase13_notebook_v2.zip", "format 2"),
            ("phase14_notebook_v3.zip", "format 3 + audio"),
        ):
            path = args.fixtures / fixture
            if not path.is_file():
                print(f"SKIP {label}: {path} not found")
                continue
            st, _, body = c.upload(
                "/backup/import", "file", fixture, path.read_bytes(), "application/zip"
            )
            check(st == 200 and b"Import complete" in body, f"import {label} archive")

        def one(sql: str) -> tuple | None:
            import sqlite3

            db = sqlite3.connect(
                f"file:{(data / 'litenotebook.sqlite3').as_posix()}?mode=ro", uri=True
            )
            try:
                return db.execute(sql).fetchone()
            finally:
                db.close()

        quiz = one("SELECT notebook_id, id FROM quizzes LIMIT 1")
        if quiz:
            st, _, _ = c.req("GET", f"/notebooks/{quiz[0]}/study/quizzes/{quiz[1]}")
            check(st == 200, "imported quiz opens")
        audio = one(
            "SELECT o.notebook_id, o.id, a.id FROM audio_overviews o "
            "JOIN audio_assets a ON a.overview_id = o.id LIMIT 1"
        )
        if audio:
            nb_a, oid, aid = audio
            st, _, text = c.req(
                "GET", f"/notebooks/{nb_a}/study/audio/{oid}/transcript.txt"
            )
            check(st == 200 and len(text) > 20, "imported audio transcript opens")
            request = urllib.request.Request(
                f"{base}/notebooks/{nb_a}/study/audio/{oid}/audio/{aid}",
                headers={"Range": "bytes=0-99"},
            )
            with urllib.request.urlopen(request, timeout=30) as r:
                check(
                    r.status == 206 and r.read(4) == b"RIFF",
                    "imported WAV plays (Range request)",
                )
        # deleting the uploaded source removes its stored file
        sid_row = one(f"SELECT id FROM sources WHERE notebook_id = {nb}")
        if sid_row:
            sid = sid_row[0]
            stored = data / "notebooks" / str(nb) / "sources" / f"{sid}.pdf"
            st, _, _ = c.req(
                "POST", f"/notebooks/{nb}/sources/{sid}/delete", {"confirm": str(sid)}
            )
            check(
                st in (200, 303) and not stored.exists(),
                "source deleted and its file removed",
            )
        # quit
        t3 = time.perf_counter()
        st, _, _ = c.req("POST", "/app/quit", {})
        check(st == 200, "Quit LiteNotebook accepted")
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        RESULTS["quit_s"] = round(time.perf_counter() - t3, 2)
        check(proc.poll() is not None, f"process exited (code {proc.returncode})")
        time.sleep(1.0)
        left = app_processes(folder, data)
        check(not left, f"no LiteNotebook process left {left}")
        # restart: same data, then quit again
        record.unlink(missing_ok=True)
        again = subprocess.Popen(
            [str(exe), "--no-browser"],
            env=env,
            cwd=str(tmp),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:

            def up() -> str | None:
                # the port may differ (the old one can still be in TIME_WAIT):
                # read the live URL from the instance lock, as a 2nd launch does
                try:
                    info = json.loads((data / ".instance.lock").read_text("utf-8"))
                    url = info.get("url") if info.get("pid") == again.pid else None
                    if url and Client(url).req("GET", "/livez")[0] == 200:
                        return str(url)
                except (OSError, ValueError):
                    pass
                return None

            url2 = wait_for(up, 60, 0.3)
            check(bool(url2), f"restarts on the same data ({url2})")
            if url2:
                st, _, body = Client(url2).req(
                    "GET", f"/notebooks/{nb}/search?q=violet"
                )
                check(b"Harbour" in body, "data still there after a restart")
                Client(url2).req("POST", "/app/quit", {})
            again.wait(timeout=30)
        finally:
            if again.poll() is None:
                again.terminate()
        log_text = (logs / "litenotebook.log").read_text("utf-8", "replace")
        check("Traceback" not in log_text, "no tracebacks in the log")
        check("Quillport trams are violet" not in log_text, "no note text in the log")
        m = re.search(r"Ready at \S+ after ([\d.]+) s", log_text)
        RESULTS["health_ready_s"] = float(m.group(1)) if m else None
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        if not args.keep:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
    RESULTS["failed"] = FAILED
    print(json.dumps(RESULTS, indent=1))
    if args.json:
        args.json.write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")
    return 1 if FAILED else 0


def _json_job(c: Client, job_id: str) -> dict | None:
    request = urllib.request.Request(
        f"{c.base}/ai/jobs/{job_id}", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as r:
        job = json.loads(r.read())
    return job if job.get("state") != "running" else None


if __name__ == "__main__":
    raise SystemExit(main())
