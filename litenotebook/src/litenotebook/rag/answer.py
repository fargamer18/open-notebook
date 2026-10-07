"""Answer a question from one notebook with a single LLM call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from litenotebook.llm.client import ChatResult, LLMClient
from litenotebook.rag.citations import CitationReport, resolve
from litenotebook.rag.prompts import NO_RESULTS_ANSWER, build_messages
from litenotebook.retrieval.hybrid import Retriever, SearchResult

ANSWER_MAX_TOKENS = 1024
ANSWER_TEMPERATURE = 0.2


@dataclass
class Answer:
    question: str
    text: str
    results: list[SearchResult]
    citations: CitationReport
    llm: ChatResult | None  # None when nothing was retrieved (no LLM call made)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def uncited(self) -> bool:
        return self.llm is not None and not self.citations.citations


def answer_question(
    question: str,
    notebook_id: int,
    retriever: Retriever,
    llm: LLMClient,
    *,
    top_k: int | None = None,
) -> Answer:
    started = time.perf_counter()
    results = retriever.search(notebook_id, question, top_k)
    retrieved = time.perf_counter()
    timings = {"retrieval": retrieved - started}

    if not results:
        timings["total"] = time.perf_counter() - started
        return Answer(
            question, NO_RESULTS_ANSWER, [], CitationReport([], []), None, timings
        )

    chat = llm.chat(
        build_messages(question, results),
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=ANSWER_TEMPERATURE,
    )
    timings["llm"] = chat.elapsed_s
    timings["total"] = time.perf_counter() - started
    return Answer(
        question, chat.content, results, resolve(chat.content, results), chat, timings
    )
