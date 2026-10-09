"""POSIX orphan guard: ``python -m litenotebook.platform_utils.supervise <argv…>``.

Used for llama-server (``processes.orphan_guard``). It runs in its own
session (so it can be stopped as a group), which also means it would outlive
LiteNotebook if LiteNotebook died without running
its shutdown code — killed, crashed, or its terminal closed (SIGHUP). This
small parent starts llama-server as its only child and watches its own stdin,
a pipe from LiteNotebook: the kernel closes that pipe when LiteNotebook exits
for any reason, and the guard then stops exactly that child (terminate, then
kill). Windows does the same with a Job Object, so the guard is not used
there.

The first stderr line is ``litenotebook-supervise: child <pid>`` so the
manager can report the real llama-server process (memory, pid).
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading

MARKER = "litenotebook-supervise: child "
GRACE_S = 10.0


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print(
            "usage: python -m litenotebook.platform_utils.supervise <program> [args…]",
            file=sys.stderr,
        )
        return 2
    # Same process group as this guard: stopping the group stops both.
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL)  # noqa: S603 - no shell
    sys.stderr.write(f"{MARKER}{child.pid}\n")
    sys.stderr.flush()

    def stop_child() -> None:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(GRACE_S)
            except subprocess.TimeoutExpired:
                child.kill()

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)

    def watch() -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass  # LiteNotebook never writes; EOF means it is gone
        except OSError:
            pass
        stop_child()

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()
    code = child.wait()
    return 128 - code if code < 0 else code  # killed by a signal: 128 + signum


if __name__ == "__main__":
    sys.exit(main())
