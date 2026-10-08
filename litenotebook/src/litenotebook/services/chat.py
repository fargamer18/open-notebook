"""One RAG chat exchange: retrieve → stream Qwen → strip thinking → cite → persist.

``ChatGeneration.events()`` is an async generator of server-sent events. The
web layer streams it to the browser; it never holds the whole answer before
sending it. Every exit path leaves the assistant message in a final state:

* finished          → complete (with citation snapshots)
* no ready sources / nothing retrieved → complete, no LLM call
* LLM/retrieval error → failed (user-facing message, retry offered)
* Stop / client disconnect / cancellation → interrupted (partial text kept)

Only one generation runs at a time in the whole app (``GenerationGuard``):
llama-server runs with ``-np 1``, so overlapping requests would just queue
inside llama.cpp and stall each other.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio
import httpx

from litenotebook.chat import (
    INTERRUPTED_MESSAGE,
    CitedChunk,
    Exchange,
    complete_message,
    end_message,
    history_turns,
)
from litenotebook.config import Settings
from litenotebook.db import connect
from litenotebook.llm.client import ChatStream, LLMError, ThinkFilter
from litenotebook.rag.citations import resolve
from litenotebook.rag.prompts import NO_RESULTS_ANSWER, build_messages

if TYPE_CHECKING:
    from litenotebook.retrieval.hybrid import SearchResult
    from litenotebook.services.embedding import SharedEmbedder
    from litenotebook.services.settings import AppSettings

log = logging.getLogger(__name__)

NO_READY_SOURCES = (
    "No ready sources are available in this notebook yet. Upload a PDF under "
    "Sources and wait until it is Ready, then ask again."
)
STOPPED_MESSAGE = "Generation stopped. Retry to generate a new answer."
EMPTY_ANSWER = "The model returned an empty answer. Retry."
BUSY_MESSAGE = (
    "Another answer is being generated. Wait for it to finish or stop it first."
)
ANSWER_MAX_TOKENS = 1024
ANSWER_TEMPERATURE = 0.2
STALE_LEASE_S = 900.0  # a lease never started or never released is reclaimed


def sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


class GenerationGuard:
    """At most one active generation per process, with a stop flag."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: int | None = None  # assistant message id
        self._since = 0.0
        self._stop: set[int] = set()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active is not None:
                if time.monotonic() - self._since < STALE_LEASE_S:
                    return False
                log.warning("Reclaiming stale generation lease %s", self._active)
            self._active, self._since = 0, time.monotonic()
            return True

    def bind(self, message_id: int) -> None:
        with self._lock:
            self._active = message_id

    def release(self, message_id: int | None = None) -> None:
        with self._lock:
            if message_id is None or self._active in (message_id, 0):
                self._active = None
            if message_id is not None:
                self._stop.discard(message_id)

    def request_stop(self, message_id: int) -> bool:
        with self._lock:
            if self._active != message_id:
                return False
            self._stop.add(message_id)
            return True

    def stop_requested(self, message_id: int) -> bool:
        with self._lock:
            return message_id in self._stop

    @property
    def active(self) -> set[int]:
        with self._lock:
            return {self._active} if self._active else set()


class _Stopped(Exception):
    pass


@dataclass
class ChatGeneration:
    """Everything one exchange needs; ``events()`` runs it."""

    settings: Settings
    llm: AppSettings
    notebook_id: int
    exchange: Exchange
    embedder: SharedEmbedder
    guard: GenerationGuard
    transport: httpx.AsyncBaseTransport | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def message_id(self) -> int:
        return self.exchange.assistant_message_id

    # --- blocking helpers (run in worker threads) ---------------------------------

    def _has_ready_sources(self) -> bool:
        conn = connect(self.settings.db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM sources WHERE notebook_id = ? AND status = 'ready' "
                "LIMIT 1",
                (self.notebook_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def _retrieve(self) -> list[SearchResult]:
        from litenotebook.index.faiss_store import FaissStore
        from litenotebook.retrieval.hybrid import Retriever

        if not self.embedder.loaded:
            started = time.perf_counter()
            self.embedder.load()
            self.timings["model_load"] = round(time.perf_counter() - started, 3)
        conn = connect(self.settings.db_path)
        try:
            started = time.perf_counter()
            retriever = Retriever(
                conn, self.settings, self.embedder, FaissStore(conn, self.settings)
            )
            results = retriever.search(
                self.notebook_id, self.exchange.question, self.llm.rag_top_k
            )
            self.timings["retrieval"] = round(time.perf_counter() - started, 3)
            return results
        finally:
            conn.close()

    def _history(self) -> list[Any]:
        conn = connect(self.settings.db_path)
        try:
            return history_turns(
                conn,
                self.exchange.conversation.id,
                self.exchange.user_message_id,
            )
        finally:
            conn.close()

    # --- the stream ----------------------------------------------------------------

    async def events(self) -> AsyncGenerator[bytes, None]:
        started = time.perf_counter()
        parts: list[str] = []
        conn = connect(self.settings.db_path)
        mid = self.message_id
        try:
            yield sse(
                "start",
                {
                    "message_id": mid,
                    "conversation_id": self.exchange.conversation.id,
                    "created_conversation": self.exchange.created_conversation,
                    "title": self.exchange.conversation.title,
                },
            )
            try:
                if not await anyio.to_thread.run_sync(self._has_ready_sources):
                    complete_message(conn, mid, NO_READY_SOURCES, model=None)
                    yield sse("done", {"message_id": mid, "status": "complete"})
                    return
                yield sse("status", {"stage": "retrieving"})
                results = await anyio.to_thread.run_sync(self._retrieve)
                history = await anyio.to_thread.run_sync(self._history)
            except Exception as e:
                message = _retrieval_error(e)
                end_message(conn, mid, "failed", message, meta=self._meta(started))
                yield sse("error", {"message_id": mid, "message": message})
                return
            if not results:
                complete_message(conn, mid, NO_RESULTS_ANSWER, model=None)
                yield sse("done", {"message_id": mid, "status": "complete"})
                return

            yield sse("status", {"stage": "generating", "sources": len(results)})
            stream = ChatStream(
                self.llm.llm_base_url,
                self.llm.llm_model,
                api_key=self.llm.llm_api_key or None,
                timeout_s=self.settings.llm_timeout_s,
                transport=self.transport,
            )
            think = ThinkFilter()
            llm_started = time.perf_counter()
            try:
                async for delta in stream.deltas(
                    build_messages(self.exchange.question, results, history),
                    max_tokens=ANSWER_MAX_TOKENS,
                    temperature=ANSWER_TEMPERATURE,
                ):
                    if "first_token" not in self.timings:
                        self.timings["first_token"] = round(
                            time.perf_counter() - llm_started, 3
                        )
                    text = think.feed(delta)
                    if text:
                        parts.append(text)
                        yield sse("delta", {"t": text})
                    if self.guard.stop_requested(mid):
                        raise _Stopped
                tail = think.flush()
                if tail:
                    parts.append(tail)
                    yield sse("delta", {"t": tail})
            except _Stopped:
                end_message(
                    conn,
                    mid,
                    "interrupted",
                    STOPPED_MESSAGE,
                    content="".join(parts).strip(),
                    model=stream.info.model or self.llm.llm_model,
                    meta=self._meta(started, stream),
                )
                yield sse("done", {"message_id": mid, "status": "interrupted"})
                return
            except LLMError as e:
                message = str(e)
                end_message(
                    conn,
                    mid,
                    "failed",
                    message,
                    content="".join(parts).strip(),
                    meta=self._meta(started, stream),
                )
                yield sse("error", {"message_id": mid, "message": message})
                return
            self.timings["generation"] = round(time.perf_counter() - llm_started, 3)

            content = "".join(parts).strip()
            if not content:
                end_message(
                    conn, mid, "failed", EMPTY_ANSWER, meta=self._meta(started, stream)
                )
                yield sse("error", {"message_id": mid, "message": EMPTY_ANSWER})
                return
            persist_started = time.perf_counter()
            report = resolve(content, results)
            cited = [
                CitedChunk(
                    label=c.label,
                    chunk_id=c.chunk_id,
                    source_id=c.source_id,
                    source_filename=c.source_filename,
                    page_no=c.page_no,
                    chunk_ord=c.ord,
                    chunk_text=results[c.label - 1].text,
                )
                for c in report.citations
            ]
            meta = self._meta(started, stream)
            meta["invalid_labels"] = report.invalid_labels
            complete_message(
                conn,
                mid,
                content,
                model=stream.info.model or self.llm.llm_model or "llm",
                citations=cited,
                meta=meta,
            )
            self.timings["persist"] = round(time.perf_counter() - persist_started, 4)
            with conn:  # record the persistence time too (best effort)
                conn.execute(
                    "UPDATE messages SET meta = ? WHERE id = ?",
                    (json.dumps({**meta, "timings": dict(self.timings)}), mid),
                )
            yield sse(
                "done",
                {"message_id": mid, "status": "complete", "citations": len(cited)},
            )
        except BaseException:
            # Client disconnect, cancellation or a bug: never leave "generating".
            try:
                end_message(
                    conn,
                    mid,
                    "interrupted",
                    INTERRUPTED_MESSAGE,
                    content="".join(parts).strip(),
                    model=self.llm.llm_model or None,
                    meta=self._meta(started),
                )
            except Exception:
                log.exception("Could not mark message %s interrupted", mid)
            raise
        finally:
            conn.close()
            self.guard.release(mid)

    def _meta(self, started: float, stream: ChatStream | None = None) -> dict[str, Any]:
        self.timings["total"] = round(time.perf_counter() - started, 3)
        meta: dict[str, Any] = {"timings": dict(self.timings)}
        if stream is not None:
            meta["finish_reason"] = stream.info.finish_reason
            if stream.info.timings:
                meta["llama_timings"] = {
                    k: v
                    for k, v in stream.info.timings.items()
                    if isinstance(v, int | float)
                }
        return meta


def _retrieval_error(error: Exception) -> str:
    from litenotebook.embed.embedder import EmbeddingModelUnavailableError

    if isinstance(error, EmbeddingModelUnavailableError):
        return str(error)
    log.exception("Retrieval failed", exc_info=error)
    return "Retrieval failed; details are in the server log. Retry."
