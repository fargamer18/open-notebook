"""Quiz generation: selected material → evidence → Qwen → validation → save.

Runs in a background thread with its own SQLite connection. Every exit
leaves the quiz in a final state (ready, failed or cancelled); a quiz is
never saved with unvalidated questions, and nothing is invented to reach
the requested count.

Pipeline::

    Retrieving evidence…   (MiniLM only when a focus is given)
    Starting Local AI…     (managed llama-server, if it is not running)
    Generating questions…  (one request per batch, sized to the context)
    Validating quiz…       (each batch; bounded repair requests for shortfalls)
    Saving…

The generation lease (``GenerationGuard``) is taken by the caller before the
thread starts and released here, so Chat and quizzes never share the single
llama-server slot at the same time.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import httpx

from litenotebook.db import connect
from litenotebook.llm.client import LLMError, LLMResponseError, StreamStopped
from litenotebook.study import store
from litenotebook.study.evidence import EvidenceError, gather
from litenotebook.study.model import KIND_PREFIX, EvidenceItem, Material, QuizQuestion
from litenotebook.study.prompts import build_messages, response_schema
from litenotebook.study.validate import (
    MalformedOutputError,
    parse_output,
    validate_batch,
)

if TYPE_CHECKING:
    from litenotebook.config import Settings
    from litenotebook.services.chat import GenerationGuard
    from litenotebook.services.embedding import SharedEmbedder
    from litenotebook.services.settings import AppSettings

log = logging.getLogger(__name__)

MAX_REPAIR_CALLS = 2
TOKENS_PER_QUESTION = 330
PROMPT_TOKENS = 900  # system prompt + instructions, rounded up
CHARS_PER_TOKEN = 3
MAX_EVIDENCE_CHARS = 9000
DEFAULT_CONTEXT = 4096
TEMPERATURE = 0.3

STAGES = {
    "queued": "Queued…",
    "loading_model": "Loading the embedding model…",
    "retrieving": "Retrieving evidence…",
    "starting_runtime": "Starting Local AI…",
    "generating": "Generating questions…",
    "validating": "Validating quiz…",
    "saving": "Saving…",
}
NONE_VALID = (
    "No question could be grounded safely in the selected material, so no quiz "
    "was saved. Try other sources, a different focus or an easier difficulty."
)


class Cancelled(Exception):
    pass


@dataclass
class QuizJob:
    """What one generation needs (built by the web layer)."""

    settings: Settings
    notebook_id: int
    quiz_id: int
    material: Material
    difficulty: str
    types: list[str]
    count: int
    llm: AppSettings
    guard: GenerationGuard
    prepare: Callable[[], AppSettings | None] | None = None
    embedder: SharedEmbedder | None = None
    managed: bool = False  # Local AI runs a llama-server with a known context
    transport: httpx.BaseTransport | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    timings: dict[str, float] = field(default_factory=dict)
    calls: int = 0

    @property
    def owner(self) -> int:
        return -self.quiz_id


def type_plan(types: Sequence[str], count: int) -> list[str]:
    """``count`` question types, cycling through the selected ones."""
    return [types[i % len(types)] for i in range(count)]


def batch_size(context: int) -> int:
    if context >= 6000:
        return 5
    if context >= 4000:
        return 4
    if context >= 3000:
        return 3
    return 2


def evidence_budget(context: int, questions: int) -> int:
    room = context - TOKENS_PER_QUESTION * questions - 150 - PROMPT_TOKENS
    return max(1500, min(MAX_EVIDENCE_CHARS, room * CHARS_PER_TOKEN))


def label_batch(items: Sequence[EvidenceItem]) -> dict[str, EvidenceItem]:
    counters: dict[str, int] = {}
    out: dict[str, EvidenceItem] = {}
    for item in items:
        counters[item.kind] = counters.get(item.kind, 0) + 1
        label = f"{KIND_PREFIX[item.kind]}{counters[item.kind]}"
        out[label] = replace(item, label=label)
    return out


def _context_size(job: QuizJob, conn: Any) -> int:
    """The model's context: the managed profile's (read after the start, which
    may have lowered it), or what llama-server reports at /props (a local
    request), else a conservative default."""
    if job.managed:
        from litenotebook.ai.store import AIStore

        return max(1024, AIStore(conn).profile().context)
    root = job.llm.llm_base_url.rstrip("/")
    root = root[: -len("/v1")] if root.endswith("/v1") else root
    try:
        with httpx.Client(timeout=3.0, transport=job.transport) as client:
            r = client.get(f"{root}/props")
            if r.status_code == 200:
                n = r.json().get("default_generation_settings", {}).get("n_ctx")
                if isinstance(n, int) and n >= 1024:
                    return n
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    return DEFAULT_CONTEXT


class QuizGenerator:
    def __init__(self, job: QuizJob) -> None:
        self.job = job
        self.conn = connect(job.settings.db_path)
        self.found: list[QuizQuestion] = []
        self.rejected: list[str] = []
        self.samples: list[dict[str, str]] = []  # for diagnostics (meta only)
        self.model: str | None = None
        self.use_schema = True
        self.started = time.perf_counter()

    # --- helpers ------------------------------------------------------------------

    def _stage(self, stage: str) -> None:
        self._check()
        store.set_stage(self.conn, self.job.quiz_id, stage)

    def _check(self) -> None:
        if self.job.cancel.is_set():
            raise Cancelled

    def _timed(self, key: str, started: float) -> None:
        self.job.timings[key] = round(
            self.job.timings.get(key, 0.0) + time.perf_counter() - started, 3
        )

    def _meta(self) -> dict[str, Any]:
        self.job.timings["total"] = round(time.perf_counter() - self.started, 3)
        rejected: dict[str, int] = {}
        for reason in self.rejected:
            rejected[reason] = rejected.get(reason, 0) + 1
        return {
            "timings": dict(self.job.timings),
            "llm_calls": self.job.calls,
            "rejected": rejected,
            "rejected_samples": self.samples,
        }

    def _ask(self, items: dict[str, EvidenceItem], plan: list[str]) -> list[Any]:
        from litenotebook.llm.client import complete_streaming

        job = self.job
        messages = build_messages(
            list(items.values()),
            plan,
            job.difficulty,
            job.material.focus,
            [q.question for q in self.found],
        )
        job.guard.touch(job.owner)
        started = time.perf_counter()
        extra: dict[str, Any] = {}
        if self.use_schema:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "quiz", "schema": response_schema(len(plan))},
            }
        kwargs: dict[str, Any] = {
            "api_key": job.llm.llm_api_key or None,
            "timeout_s": job.settings.llm_timeout_s,
            "max_tokens": TOKENS_PER_QUESTION * len(plan) + 150,
            "temperature": TEMPERATURE,
            "transport": job.transport,
            "should_stop": job.cancel.is_set,
        }
        try:
            job.calls += 1
            text, info = complete_streaming(
                job.llm.llm_base_url, job.llm.llm_model, messages, extra=extra, **kwargs
            )
        except LLMResponseError as e:
            # Some OpenAI-compatible servers reject json_schema: retry plain once.
            if not self.use_schema or "HTTP 4" not in str(e):
                raise
            self.use_schema = False
            job.calls += 1
            text, info = complete_streaming(
                job.llm.llm_base_url, job.llm.llm_model, messages, **kwargs
            )
        finally:
            self._timed("generation", started)
        self.model = info.model or job.llm.llm_model or self.model
        try:
            return parse_output(text)
        except MalformedOutputError:
            self.rejected.append("malformed reply")
            return []

    def _validate(
        self, raw: list[Any], items: dict[str, EvidenceItem], limit: int
    ) -> None:
        started = time.perf_counter()
        kept, rejected = validate_batch(
            raw,
            items,
            allowed_types=self.job.types,
            allowed_sources=self.job.material.source_ids,
            accepted=self.found,
            difficulty=self.job.difficulty,
            limit=limit,
        )
        self.found.extend(kept)
        self.rejected.extend(r.reason for r in rejected)
        for r in rejected[: max(0, 12 - len(self.samples))]:
            item = raw[r.index] if r.index < len(raw) else None
            text = item.get("question") if isinstance(item, dict) else None
            self.samples.append({"reason": r.reason, "question": str(text or "")[:160]})
        missing = limit - len(kept) - len(rejected)
        self.rejected.extend(["not returned"] * max(0, missing))
        self._timed("validation", started)

    # --- the run ------------------------------------------------------------------

    def run(self) -> None:
        job = self.job
        try:
            self._run()
        except (Cancelled, StreamStopped):
            store.finish_failed(
                self.conn,
                job.quiz_id,
                "Cancelled.",
                status="cancelled",
                meta=self._meta(),
            )
        except EvidenceError as e:
            store.finish_failed(self.conn, job.quiz_id, str(e), meta=self._meta())
        except LLMError as e:
            store.finish_failed(self.conn, job.quiz_id, str(e), meta=self._meta())
        except Exception as e:
            log.exception("Quiz %s generation failed", job.quiz_id)
            message = (
                str(e) if type(e).__name__ in ("SetupError", "FallbackError") else ""
            )
            store.finish_failed(
                self.conn,
                job.quiz_id,
                message or "Quiz generation failed; details are in the server log.",
                meta=self._meta(),
            )
        finally:
            job.guard.release(job.owner)
            self.conn.close()

    def _run(self) -> None:
        job = self.job
        focus = job.material.focus.strip()
        embedder = job.embedder if focus else None  # MiniLM only for a focus
        if embedder is not None and not embedder.loaded:
            self._stage("loading_model")
            started = time.perf_counter()
            embedder.load()
            self._timed("model_load", started)
        self._stage("retrieving")

        started = time.perf_counter()
        per_batch = batch_size(DEFAULT_CONTEXT)
        batches = -(-job.count // per_batch)
        pool = gather(
            self.conn,
            job.settings,
            job.notebook_id,
            job.material,
            wanted=max(4, batches * 5),
            embedder=embedder,
        )
        self._timed("retrieval", started)
        self._check()

        if job.prepare is not None:
            self._stage("starting_runtime")
            started = time.perf_counter()
            ready = job.prepare()
            if ready is not None:
                job.llm = ready
            self._timed("runtime_start", started)
        context = _context_size(job, self.conn)
        per_batch = batch_size(context)
        budget = evidence_budget(context, per_batch)

        plan = type_plan(job.types, job.count)
        uses = [0] * len(pool.items)

        def next_items() -> dict[str, EvidenceItem]:
            """Least-used passages first, within the character budget."""
            order = sorted(range(len(pool.items)), key=lambda i: (uses[i], i))
            chosen: list[int] = []
            size = 0
            for i in order:
                length = len(pool.items[i].text) + 80
                if chosen and size + length > budget:
                    continue
                chosen.append(i)
                size += length
                if size >= budget:
                    break
            for i in chosen:
                uses[i] += 1
            chosen.sort()
            return label_batch([pool.items[i] for i in chosen])

        total_batches = -(-len(plan) // per_batch)
        for n in range(total_batches):
            part = plan[n * per_batch : (n + 1) * per_batch]
            self._stage("generating")
            store.set_stage(
                self.conn, job.quiz_id, f"generating:{n + 1}:{total_batches}"
            )
            items = next_items()
            raw = self._ask(items, part)
            self._stage("validating")
            self._validate(raw, items, len(part))

        repairs = 0
        while len(self.found) < job.count and repairs < MAX_REPAIR_CALLS:
            self._check()
            missing = job.count - len(self.found)
            part = type_plan(job.types[repairs % len(job.types) :] + job.types, missing)
            part = part[:per_batch]
            repairs += 1
            self._stage("generating")
            store.set_stage(self.conn, job.quiz_id, "generating:repair")
            items = next_items()
            before = len(self.rejected)
            raw = self._ask(items, part)
            self._stage("validating")
            self._validate(raw, items, len(part))
            # replacement requests are not counted as rejections of the quiz
            del self.rejected[before:]

        self._check()
        if not self.found:
            store.finish_failed(self.conn, job.quiz_id, NONE_VALID, meta=self._meta())
            return
        self._stage("saving")
        started = time.perf_counter()
        message = None
        if len(self.found) < job.count:
            short = job.count - len(self.found)
            message = (
                f"Generated {len(self.found)} of {job.count} requested questions. "
                + (
                    f"{short} questions were left out because they"
                    if short != 1
                    else "1 question was left out because it"
                )
                + " could not be grounded safely in the selected material."
            )
        meta = self._meta()
        meta["semantic"] = pool.semantic
        meta["evidence_items"] = len(pool.items)
        meta["context"] = context
        meta["retrieval_detail"] = pool.timings
        store.save_questions(
            self.conn,
            job.quiz_id,
            self.found,
            message=message,
            model=self.model,
            meta=meta,
        )
        meta["timings"]["save"] = round(time.perf_counter() - started, 4)
        with self.conn:
            self.conn.execute(
                "UPDATE quizzes SET meta = ? WHERE id = ?",
                (json.dumps(meta), job.quiz_id),
            )


def run_job(job: QuizJob) -> None:
    QuizGenerator(job).run()
