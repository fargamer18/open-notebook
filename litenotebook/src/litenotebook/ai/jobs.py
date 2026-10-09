"""Background jobs for long setup work (downloads, installs, tests).

Each job runs in a thread, reports a stage and real byte counts (no invented
percentages), can be cancelled, and keeps its final result or a specific
error message. The web UI polls a job; the CLI runs the same functions
synchronously.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from litenotebook.ai.download import DownloadCancelled, Progress


@dataclass
class Job:
    id: str
    kind: str
    label: str
    state: str = "running"  # running | done | failed | cancelled
    stage: str = "Preparing…"
    done: int | None = None
    total: int | None = None
    rate_bps: float | None = None
    message: str = ""
    result: Any = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def update(
        self, stage: str | None = None, progress: Progress | None = None
    ) -> None:
        if stage:
            self.stage = stage
        if progress is not None:
            self.done, self.total, self.rate_bps = (
                progress.done,
                progress.total,
                progress.rate_bps,
            )

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ("cancel", "result")}


class JobManager:
    def __init__(self, keep: int = 20) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.keep = keep

    def start(self, kind: str, label: str, work: Callable[[Job], Any]) -> Job:
        with self._lock:
            running = next(
                (
                    j
                    for j in self._jobs.values()
                    if j.kind == kind and j.state == "running"
                ),
                None,
            )
            if running is not None:
                return running  # one job per kind; repeated clicks attach to it
            job = Job(uuid.uuid4().hex[:16], kind, label)
            self._jobs[job.id] = job
            for old in sorted(self._jobs.values(), key=lambda j: j.started)[
                : -self.keep
            ]:
                if old.state != "running":
                    self._jobs.pop(old.id, None)

        def run() -> None:
            try:
                job.result = work(job)
                job.state, job.stage = "done", "Complete"
            except DownloadCancelled:
                job.state, job.message = "cancelled", "Cancelled"
            except Exception as e:  # shown to the user: messages are written for it
                job.state = "cancelled" if job.cancel.is_set() else "failed"
                job.message = str(e) or type(e).__name__
            finally:
                job.finished = time.time()

        threading.Thread(target=run, name=f"ai-job-{kind}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self, kind: str | None = None) -> list[Job]:
        return [
            j
            for j in self._jobs.values()
            if j.state == "running" and (kind is None or j.kind == kind)
        ]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.state != "running":
            return False
        job.cancel.set()
        return True
