"""Audio Overview script generation: evidence → outline → script → save.

Runs in a background thread with its own SQLite connection, holding the
same single generation lease as Chat and quizzes. Bounded model calls:

    1 outline request (+1 retry if it yields no usable section)
    1–4 script requests (sections grouped so each reply stays ~800 words)
    +1 repair request at most, for a script group that yielded nothing

Every exit leaves the overview final: script_ready with a validated,
immutable transcript, or failed / cancelled with no transcript.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from litenotebook.db import connect
from litenotebook.llm.client import LLMError, LLMResponseError, StreamStopped
from litenotebook.study import audio_store as store
from litenotebook.study.audio_model import (
    SECTIONS_FOR,
    OutlineSection,
    Script,
    ScriptSection,
    estimate_seconds,
    target_words,
)
from litenotebook.study.audio_prompts import (
    outline_messages,
    outline_schema,
    script_messages,
    script_schema,
)
from litenotebook.study.audio_validate import (
    Evidence,
    one_sided,
    validate_outline,
    validate_script,
)
from litenotebook.study.evidence import EvidenceError, gather
from litenotebook.study.generate import Cancelled, _context_size, label_batch
from litenotebook.study.model import EvidenceItem, Material
from litenotebook.study.validate import MalformedOutputError, parse_output

if TYPE_CHECKING:
    from litenotebook.config import Settings
    from litenotebook.services.chat import GenerationGuard
    from litenotebook.services.embedding import SharedEmbedder
    from litenotebook.services.settings import AppSettings

log = logging.getLogger(__name__)

WORDS_PER_CALL = 800
TOKENS_PER_WORD = 1.6  # JSON and labels included, rounded up
OUTLINE_TOKENS = 900
PROMPT_TOKENS = 1100
CHARS_PER_TOKEN = 3
MAX_EVIDENCE_CHARS = 12000
TEMPERATURE = 0.5
OWNER_BASE = 1_000_000_000  # guard owner ids: quizzes -id, audio -(base + id)

NOTHING_GROUNDED = (
    "No part of the script could be grounded safely in the selected material, so "
    "no transcript was saved. Try other sources or a different focus."
)


@dataclass
class AudioJob:
    settings: Settings
    notebook_id: int
    overview_id: int
    material: Material
    fmt: str
    minutes: int
    wpm: int
    llm: AppSettings
    guard: GenerationGuard
    prepare: Callable[[], AppSettings | None] | None = None
    embedder: SharedEmbedder | None = None
    managed: bool = False
    transport: httpx.BaseTransport | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    timings: dict[str, float] = field(default_factory=dict)
    calls: int = 0

    @property
    def owner(self) -> int:
        return -(OWNER_BASE + self.overview_id)


def _split(sections: list[OutlineSection], words: int) -> list[list[OutlineSection]]:
    per_section = max(1, words // max(1, len(sections)))
    per_call = max(1, WORDS_PER_CALL // per_section)
    return [sections[i : i + per_call] for i in range(0, len(sections), per_call)]


class AudioGenerator:
    def __init__(self, job: AudioJob) -> None:
        self.job = job
        self.conn = connect(job.settings.db_path)
        self.model: str | None = None
        self.use_schema = True
        self.removed: list[dict[str, str]] = []
        self.problems: list[str] = []
        self.started = time.perf_counter()

    def _check(self) -> None:
        if self.job.cancel.is_set():
            raise Cancelled

    def _stage(self, status: str, stage: str | None = None) -> None:
        self._check()
        store.set_stage(self.conn, self.job.overview_id, status, stage)

    def _timed(self, key: str, started: float) -> None:
        self.job.timings[key] = round(
            self.job.timings.get(key, 0.0) + time.perf_counter() - started, 3
        )

    def _meta(self) -> dict[str, Any]:
        self.job.timings["total"] = round(time.perf_counter() - self.started, 3)
        reasons: dict[str, int] = {}
        for r in self.removed:
            key = r["reason"].split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
        return {
            "timings": dict(self.job.timings),
            "llm_calls": self.job.calls,
            "removed": reasons,
            "removed_samples": self.removed[:12],
            "problems": self.problems[:12],
        }

    def _ask(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        max_tokens: int,
        timer: str,
    ) -> Any:
        from litenotebook.llm.client import complete_streaming

        job = self.job
        job.guard.touch(job.owner)
        extra: dict[str, Any] = {}
        if self.use_schema:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "audio", "schema": schema},
            }
        kwargs: dict[str, Any] = {
            "api_key": job.llm.llm_api_key or None,
            "timeout_s": max(job.settings.llm_timeout_s, 600.0),
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "transport": job.transport,
            "should_stop": job.cancel.is_set,
        }
        started = time.perf_counter()
        try:
            job.calls += 1
            text, info = complete_streaming(
                job.llm.llm_base_url, job.llm.llm_model, messages, extra=extra, **kwargs
            )
        except LLMResponseError as e:
            if not self.use_schema or "HTTP 4" not in str(e):
                raise
            self.use_schema = False
            job.calls += 1
            text, info = complete_streaming(
                job.llm.llm_base_url, job.llm.llm_model, messages, **kwargs
            )
        finally:
            self._timed(timer, started)
        self.model = info.model or job.llm.llm_model or self.model
        try:
            return json.loads(text)
        except ValueError:
            try:  # truncated: keep the complete objects
                return {"sections": parse_output(text)}
            except MalformedOutputError:
                self.problems.append(f"{timer}: malformed reply")
                return {}

    def run(self) -> None:
        job = self.job
        try:
            self._run()
        except (Cancelled, StreamStopped):
            store.fail(
                self.conn,
                job.overview_id,
                "Cancelled.",
                status="cancelled",
                meta=self._meta(),
            )
        except (EvidenceError, LLMError) as e:
            store.fail(self.conn, job.overview_id, str(e), meta=self._meta())
        except Exception as e:
            log.exception("Audio overview %s generation failed", job.overview_id)
            message = (
                str(e) if type(e).__name__ in ("SetupError", "FallbackError") else ""
            )
            store.fail(
                self.conn,
                job.overview_id,
                message or "Script generation failed; details are in the server log.",
                meta=self._meta(),
            )
        finally:
            job.guard.release(job.owner)
            self.conn.close()

    def _run(self) -> None:
        job = self.job
        focus = job.material.focus.strip()
        embedder = job.embedder if focus else None
        self._stage("retrieving")
        if embedder is not None and not embedder.loaded:
            store.set_stage(self.conn, job.overview_id, "retrieving", "loading_model")
            started = time.perf_counter()
            embedder.load()
            self._timed("model_load", started)
            self._stage("retrieving")
        sections_wanted = SECTIONS_FOR[job.minutes]
        started = time.perf_counter()
        pool = gather(
            self.conn,
            job.settings,
            job.notebook_id,
            job.material,
            wanted=sections_wanted * 3,
            embedder=embedder,
        )
        self._timed("retrieval", started)

        if job.prepare is not None:
            store.set_stage(
                self.conn, job.overview_id, "retrieving", "starting_runtime"
            )
            started = time.perf_counter()
            ready = job.prepare()
            if ready is not None:
                job.llm = ready
            self._timed("runtime_start", started)
        context = _context_size(job, self.conn)  # type: ignore[arg-type]
        budget = max(
            2000,
            min(
                MAX_EVIDENCE_CHARS,
                (context - OUTLINE_TOKENS - PROMPT_TOKENS) * CHARS_PER_TOKEN,
            ),
        )
        chosen: list[EvidenceItem] = []
        size = 0
        for item in pool.items:
            if chosen and size + len(item.text) + 80 > budget:
                continue
            chosen.append(item)
            size += len(item.text) + 80
        package = label_batch(chosen)
        evidence = Evidence(package, job.material.source_ids)

        # Stage 1: outline
        self._stage("generating_script", "outline")
        outline: list[OutlineSection] = []
        title = ""
        for _attempt in range(2):
            raw = self._ask(
                outline_messages(
                    list(package.values()), job.fmt, sections_wanted, focus
                ),
                outline_schema(sections_wanted),
                OUTLINE_TOKENS,
                "outline",
            )
            started = time.perf_counter()
            title, outline, problems = validate_outline(raw, evidence, sections_wanted)
            self.problems.extend(problems)
            self._timed("validation", started)
            if outline:
                break
            self._check()
        if not outline:
            store.fail(self.conn, job.overview_id, NOTHING_GROUNDED, meta=self._meta())
            return

        # Stage 2: script, in bounded groups of sections
        words = target_words(job.minutes, job.wpm)
        groups = _split(outline, words)
        per_section = max(60, words // len(outline))
        result: list[ScriptSection] = []
        repairs = 0
        for n, group in enumerate(groups):
            store.set_stage(
                self.conn,
                job.overview_id,
                "generating_script",
                f"script:{n + 1}:{len(groups)}",
            )
            labels = [r for s in group for r in s.refs]
            items = [package[label] for label in dict.fromkeys(labels)]
            group_words = per_section * len(group)
            attempts = 0
            while True:
                self._check()
                raw = self._ask(
                    script_messages(
                        items,
                        job.fmt,
                        group,
                        group_words,
                        first=n == 0,
                        last=n == len(groups) - 1,
                        done_titles=[s.title for s in result],
                        focus=focus,
                    ),
                    script_schema(job.fmt, len(group)),
                    int(group_words * TOKENS_PER_WORD) + 300,
                    "script",
                )
                started = time.perf_counter()
                raw_sections = raw.get("sections") if isinstance(raw, dict) else None
                kept, removed = validate_script(
                    raw_sections if isinstance(raw_sections, list) else [],
                    evidence,
                    job.fmt,
                    [s.title for s in group],
                )
                self.removed.extend(removed)
                self._timed("validation", started)
                if kept or repairs >= 1 or attempts >= 1:
                    break
                repairs += 1
                attempts += 1
            result.extend(kept)
        self._check()
        if not result:
            store.fail(self.conn, job.overview_id, NOTHING_GROUNDED, meta=self._meta())
            return
        if one_sided(result, job.fmt):
            self.problems.append("one host barely speaks")

        used = {r for s in result for r in s.refs}
        script = Script(
            title=self._title(title),
            sections=result,
            evidence={label: package[label] for label in used},
        )
        message = self._message(script.words, words)
        meta = self._meta()
        meta.update(
            context=context,
            evidence_items=len(package),
            outline_sections=len(outline),
            target_words=words,
            semantic=pool.semantic,
        )
        started = time.perf_counter()
        store.save_script(
            self.conn,
            job.overview_id,
            script,
            message=message,
            model=self.model,
            meta=meta,
        )
        meta["timings"]["save"] = round(time.perf_counter() - started, 4)
        with self.conn:
            self.conn.execute(
                "UPDATE audio_overviews SET meta = ? WHERE id = ?",
                (json.dumps(meta), job.overview_id),
            )

    def _title(self, outline_title: str) -> str:
        """The user's title or the deterministic default chosen at creation.
        The model's suggestion is kept in meta only (it is untrusted text)."""
        if outline_title:
            self.problems.append(f"outline title: {outline_title[:120]}")
        row = self.conn.execute(
            "SELECT title FROM audio_overviews WHERE id = ?", (self.job.overview_id,)
        ).fetchone()
        return str(row[0]) if row else "Audio overview"

    def _message(self, got: int, wanted: int) -> str | None:
        notes = []
        if got < wanted * 0.6:
            notes.append(
                f"The script is shorter than requested (about "
                f"{math.ceil(estimate_seconds(got, self.job.wpm) / 60)} min of the "
                f"~{self.job.minutes} min target)."
            )
        if self.removed:
            n = len(self.removed)
            notes.append(
                f"{n} passage{'s were' if n != 1 else ' was'} left out because "
                f"{'they' if n != 1 else 'it'} could not be grounded safely in the "
                "selected material."
            )
        return " ".join(notes) or None


def run_job(job: AudioJob) -> None:
    AudioGenerator(job).run()
