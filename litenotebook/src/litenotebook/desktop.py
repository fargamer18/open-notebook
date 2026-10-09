"""Desktop launch: what happens when the user starts LiteNotebook.

    LiteNotebook (no arguments)
      → per-user log file
      → is LiteNotebook already running for this data folder?
           yes: open its page (URL from the instance lock) and exit
      → choose a loopback port (the configured one, else a free one;
        nothing that already listens is ever stopped)
      → start the web app on 127.0.0.1 (it takes the data-folder lock)
      → wait until /healthz answers, then open the default browser
      → run until "Quit LiteNotebook" (Settings), Ctrl+C or log-out;
        on exit the app stops only the processes it started

If anything fails before the page opens, the reason goes to the log file
and to the user: a native message box on Windows, otherwise a small page
opened in the browser (no terminal needed).

``entry()`` is the packaged executable's main: ``--internal <module>`` runs a
helper (the POSIX orphan guard), a command such as ``serve`` or ``notebook``
runs the CLI, and no command (or only desktop options) starts the app.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import logging
import os
import runpy
import socket
import sys
import threading
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from litenotebook import __version__
from litenotebook.config import Settings, load_settings
from litenotebook.platform_utils.filesystem import InstanceLock, in_use
from litenotebook.platform_utils.system import (
    INTERNAL_FLAG,
    INTERNAL_MODULES,
    open_browser,
    os_name,
    show_error_dialog,
)

log = logging.getLogger("litenotebook.desktop")
HEALTH_TIMEOUT_S = 90.0
DESKTOP_FLAGS = {"--no-browser", "--port", "--data-dir", "--verbose", "-v"}
DESKTOP_ACTIONS = {"--install-desktop-entry", "--uninstall-desktop-entry", "--version"}


class DesktopError(RuntimeError):
    """A start-up problem with a message for the user."""


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def choose_port(host: str, preferred: int) -> int:
    """The preferred port if nothing listens there, else a free one chosen by
    the OS. Never touches whatever already uses the preferred port."""
    if port_is_free(host, preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def is_litenotebook(url: str, timeout: float = 2.0) -> bool:
    """Does a LiteNotebook answer at this loopback URL?"""
    try:
        with urllib.request.urlopen(f"{url}/livez", timeout=timeout) as r:
            data = json.loads(r.read(65536))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("app") == "LiteNotebook"


def existing_instance(settings: Settings) -> str | None:
    """URL of a LiteNotebook already serving this data folder, if any."""
    if not in_use(settings.data_dir):
        return None
    holder = InstanceLock(settings.data_dir).holder() or {}
    url = holder.get("url")
    if isinstance(url, str) and url.startswith("http://127.0.0.1:"):
        return url
    return ""  # in use, but by a command or an older version: no page to open


def _startup_error_page(
    settings: Settings, message: str, log_file: Path | None
) -> Path:
    folder = settings.log_path
    folder.mkdir(parents=True, exist_ok=True)
    page = folder / "startup-error.html"
    where = html.escape(str(log_file)) if log_file else "the log folder"
    page.write_text(
        "<!doctype html><meta charset='utf-8'><title>LiteNotebook could not start"
        "</title><body style='font-family:system-ui;max-width:40rem;margin:3rem auto'>"
        "<h1>LiteNotebook could not start</h1>"
        f"<p>{html.escape(message)}</p>"
        f"<p>Details are in the log file: <code>{where}</code></p></body>",
        encoding="utf-8",
    )
    return page


def report_startup_error(
    settings: Settings, message: str, log_file: Path | None
) -> None:
    log.error("Start-up failed: %s", message)
    if sys.stderr is not None:
        with contextlib.suppress(OSError, ValueError):
            print(f"LiteNotebook could not start: {message}", file=sys.stderr)
    text = f"{message}\n\nDetails: {log_file or settings.log_path}"
    if show_error_dialog("LiteNotebook could not start", text):
        return
    with contextlib.suppress(OSError):
        open_browser(_startup_error_page(settings, message, log_file).as_uri())


def run(
    argv: Sequence[str] = (),
    *,
    settings: Settings | None = None,
) -> int:
    """Start the desktop app. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog="litenotebook", add_help=True)
    parser.add_argument(
        "--no-browser", action="store_true", help="start without opening the browser"
    )
    parser.add_argument("--port", type=int, help="preferred loopback port")
    parser.add_argument("--data-dir", type=Path, help="override the data folder")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv))
    settings = settings or load_settings()
    if args.data_dir:
        settings = replace(settings, data_dir=args.data_dir.expanduser())
    if args.port:
        settings = replace(settings, port=args.port)
    from litenotebook.applog import configure_file_logging

    log_file: Path | None = None
    started = time.perf_counter()
    try:
        log_file = configure_file_logging(settings, verbose=args.verbose)
        from litenotebook.packaging import is_frozen

        log.info(
            "LiteNotebook %s starting (%s, %s); data folder %s",
            __version__,
            "packaged" if is_frozen() else "source",
            os_name(),
            settings.data_dir,
        )
        url = existing_instance(settings)
        if url:
            if is_litenotebook(url):
                log.info("Already running at %s: opening it", url)
                if not args.no_browser:
                    open_browser(url)
                return 0
        if url is not None:
            raise DesktopError(
                "LiteNotebook is already running for this data folder but is not "
                "answering (or a LiteNotebook command is using it). Wait a moment "
                "and try again, or close the other LiteNotebook first."
            )
        return _serve(settings, args.no_browser, started)
    except DesktopError as e:
        report_startup_error(settings, str(e), log_file)
        return 1
    except Exception as e:  # anything else: never fail silently
        log.exception("Unexpected start-up failure")
        report_startup_error(settings, f"{type(e).__name__}: {e}", log_file)
        return 1


def _serve(settings: Settings, no_browser: bool, started: float) -> int:
    import uvicorn

    from litenotebook.web.app import create_app

    host = "127.0.0.1"  # the desktop app never listens beyond this computer
    port = choose_port(host, settings.port)
    if port != settings.port:
        log.info("Port %s is in use by another program; using %s", settings.port, port)
    settings = replace(settings, host=host, port=port)
    url = f"http://{host}:{port}"
    app = create_app(settings, instance_info={"url": url, "launcher": "desktop"})
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,  # our file handler
        access_log=False,  # request URLs can carry search text
        log_level="warning",
        proxy_headers=False,
        server_header=False,
    )
    server = uvicorn.Server(config)
    app.state.shutdown = lambda: setattr(server, "should_exit", True)
    ready = threading.Event()

    def open_when_ready() -> None:
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        while time.monotonic() < deadline and not server.should_exit:
            if server.started and is_litenotebook(url, timeout=1.0):
                ready.set()
                log.info("Ready at %s after %.2f s", url, time.perf_counter() - started)
                if not no_browser:
                    open_browser(url)
                    log.info(
                        "Browser opened after %.2f s", time.perf_counter() - started
                    )
                return
            time.sleep(0.1)

    threading.Thread(target=open_when_ready, name="open-browser", daemon=True).start()
    server.run()
    if not ready.is_set():
        other = existing_instance(settings)
        if other and is_litenotebook(other):  # lost a race with another launch
            if not no_browser:
                open_browser(other)
            return 0
        raise DesktopError(
            "The LiteNotebook server stopped before it was ready. The log file "
            "has the details."
        )
    log.info("LiteNotebook stopped")
    return 0


# --- Linux desktop integration (portable tarball) ----------------------------------

DESKTOP_FILE = """[Desktop Entry]
Type=Application
Name=LiteNotebook
GenericName=Notebook research assistant
Comment=Local notebook for your PDFs with source-grounded answers
Exec={exe}
Icon={icon}
Terminal=false
Categories=Office;Education;Science;
StartupNotify=true
"""


def _applications_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME", "").strip()
    root = (
        Path(base)
        if base and Path(base).is_absolute()
        else Path.home() / ".local" / "share"
    )
    return root / "applications"


def install_desktop_entry(executable: Path, icon: Path) -> Path:
    """Add LiteNotebook to the user's application menu (Linux, per user)."""
    if os_name() != "linux":
        raise DesktopError("Desktop entries are for Linux; Windows uses the installer")
    folder = _applications_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "litenotebook.desktop"
    exe = str(executable).replace("\\", "\\\\").replace('"', '\\"')
    target.write_text(DESKTOP_FILE.format(exe=f'"{exe}"', icon=icon), encoding="utf-8")
    return target


def uninstall_desktop_entry() -> bool:
    target = _applications_dir() / "litenotebook.desktop"
    if target.is_file():
        target.unlink()
        return True
    return False


def _icon_for_entry() -> Path:
    from litenotebook.packaging import app_dir

    folder = app_dir()
    if folder is not None and (folder / "litenotebook.png").is_file():
        return folder / "litenotebook.png"
    return Path(__file__).parent / "web" / "static" / "icon.svg"


# --- packaged entry point -----------------------------------------------------------


def _ensure_streams() -> None:
    """A windowed (no console) Windows build has no stdout/stderr; give print()
    somewhere harmless to go."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def entry(argv: Sequence[str]) -> int:
    argv = list(argv)
    _ensure_streams()
    if argv[:1] == [INTERNAL_FLAG]:
        if len(argv) < 2 or argv[1] not in INTERNAL_MODULES:
            return 2
        sys.argv = [argv[1], *argv[2:]]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        return 0
    if argv[:1] == ["--version"]:
        print(f"LiteNotebook {__version__}")
        return 0
    if argv[:1] == ["--install-desktop-entry"]:
        target = install_desktop_entry(
            Path(sys.executable).resolve(), _icon_for_entry()
        )
        print(f"Added to the application menu: {target}")
        return 0
    if argv[:1] == ["--uninstall-desktop-entry"]:
        print("Removed" if uninstall_desktop_entry() else "Nothing to remove")
        return 0
    if not argv or argv[0].split("=")[0] in DESKTOP_FLAGS:
        return run(argv)
    from litenotebook.cli import main as cli_main

    return cli_main(argv)
