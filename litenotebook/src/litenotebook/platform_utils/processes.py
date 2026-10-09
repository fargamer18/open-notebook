"""Child processes that LiteNotebook owns, started and stopped portably.

Only processes LiteNotebook itself started are ever signalled, through the
exact ``subprocess.Popen`` handle (never by name or by a PID read from
elsewhere), so an unrelated program can never be hit.

Spawning (``spawn``):

* commands are argument lists, never ``shell=True``;
* Windows: ``CREATE_NO_WINDOW`` (no console window flashes up, also in a
  future packaged GUI build) and ``CREATE_NEW_PROCESS_GROUP`` (a Ctrl+C in
  the parent's console is not delivered to the child); the child is also put
  in a Job Object with *kill on job close*, so if LiteNotebook crashes or is
  killed, Windows terminates the child instead of leaving an orphan
  ``python.exe``;
* POSIX: with ``tree=True`` the child gets its own session/process group, so
  its descendants can be stopped together (used for future runtimes; the
  embedding worker starts no children and does not need it).

Stopping (``stop``) is always: ask nicely (``graceful`` callback, e.g. a
protocol "shutdown" or closing stdin) → wait → terminate (SIGTERM /
TerminateProcess) → wait → kill (SIGKILL / TerminateJobObject) → reap.
A POSIX process group is signalled only while its leader is alive and not
yet reaped, so a recycled PID can never be targeted.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO, Any

from litenotebook.platform_utils.system import os_name

log = logging.getLogger(__name__)

IS_WINDOWS = os_name() == "windows"
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


def creation_flags(*, system: str | None = None) -> int:
    """Popen ``creationflags`` for a background child (0 outside Windows)."""
    if os_name(system) != "windows":
        return 0
    return CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP


@dataclass
class OwnedProcess:
    proc: subprocess.Popen[bytes]
    tree: bool = False
    job: Any = None  # Windows Job Object handle, if one could be created
    outcome: str = field(default="running")

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def stdin(self) -> IO[bytes] | None:
        return self.proc.stdin

    @property
    def stdout(self) -> IO[bytes] | None:
        return self.proc.stdout

    @property
    def stderr(self) -> IO[bytes] | None:
        return self.proc.stderr

    def alive(self) -> bool:
        return self.proc.poll() is None


def _windows_job_for(proc: subprocess.Popen[bytes]) -> Any:
    """Create a kill-on-close Job Object and put ``proc`` in it. Best effort:
    failure only loses the crash-cleanup guarantee, never the process."""
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                (n, ctypes.c_ulonglong)
                for n in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ) or not kernel32.AssignProcessToJobObject(job, int(proc._handle)):  # type: ignore[attr-defined]
            kernel32.CloseHandle(job)
            return None
        return job
    except (AttributeError, OSError, ValueError) as e:
        log.debug("No Job Object for child %s: %s", proc.pid, e)
        return None


def _close_job(job: Any, *, kill: bool) -> None:
    if job is None:
        return
    with contextlib.suppress(AttributeError, OSError):
        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        if kill:
            kernel32.TerminateJobObject(job, 1)
        kernel32.CloseHandle(job)


def spawn(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    pipes: bool = True,
    tree: bool = False,
) -> OwnedProcess:
    """Start an owned background child (stdin/stdout/stderr as pipes)."""
    if not command or not isinstance(command, list | tuple):
        raise ValueError("command must be a non-empty argument list")
    pipe = subprocess.PIPE if pipes else subprocess.DEVNULL
    kwargs: dict[str, Any] = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = creation_flags()
    elif tree:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(  # noqa: S603 - argument list, no shell
        list(command),
        stdin=pipe,
        stdout=pipe,
        stderr=pipe,
        env=dict(env) if env is not None else None,
        close_fds=True,
        **kwargs,
    )
    job = _windows_job_for(proc) if IS_WINDOWS else None
    return OwnedProcess(proc, tree=tree, job=job)


def orphan_guard(command: Sequence[str]) -> list[str]:
    """Wrap a long-lived child so it never outlives LiteNotebook.

    POSIX: run it under ``platform_utils.supervise``, which stops the child
    when the pipe from this process closes (kill, crash, closed terminal).
    Windows: unchanged — ``spawn`` puts children in a kill-on-close Job
    Object. Packaged builds without an interpreter run the command as is."""
    if IS_WINDOWS:
        return list(command)
    from litenotebook.platform_utils.system import python_module_command

    try:
        return [
            *python_module_command("litenotebook.platform_utils.supervise"),
            *command,
        ]
    except RuntimeError:
        return list(command)


def _signal_tree(owned: OwnedProcess, sig: int) -> bool:
    """POSIX: signal our own process group (only if it really is ours)."""
    # returncode (not poll(), which would reap): an unreaped child — even a
    # zombie — still pins its pid and process-group id, so this is our group.
    if IS_WINDOWS or not owned.tree or owned.proc.returncode is not None:
        return False
    try:
        if os.getpgid(owned.pid) != owned.pid:  # not the group we created
            return False
        os.killpg(owned.pid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop(
    owned: OwnedProcess,
    *,
    graceful: Callable[[], None] | None = None,
    grace_s: float = 5.0,
) -> str:
    """Stop and reap. Returns "exited", "terminated" or "killed"."""
    proc = owned.proc
    if proc.poll() is None and graceful is not None:
        with contextlib.suppress(Exception):
            graceful()
    with contextlib.suppress(OSError, ValueError):
        if proc.stdin is not None:
            proc.stdin.close()  # EOF is the worker's exit signal on every OS
    outcome = "exited"
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        outcome = "terminated"
        if not _signal_tree(owned, signal.SIGTERM):
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()  # SIGTERM / TerminateProcess
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            outcome = "killed"
            if IS_WINDOWS:
                _close_job(owned.job, kill=True)  # the child and its descendants
                owned.job = None
            else:
                # group first, while the unreaped leader still pins the pgid
                _signal_tree(owned, getattr(signal, "SIGKILL", signal.SIGTERM))
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()
            proc.wait()
    _close_job(owned.job, kill=False)  # kill-on-close also ends descendants
    owned.job = None
    for stream in (proc.stdout, proc.stderr):
        with contextlib.suppress(OSError, ValueError):
            if stream is not None:
                stream.close()
    owned.outcome = outcome
    return outcome
