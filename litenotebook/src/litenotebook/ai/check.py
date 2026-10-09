"""Configuration test and a short benchmark — never with user documents.

A fixed, harmless prompt is sent; the reply must be non-empty. Measured
values (load time, time to first token, tokens per second, memory) are what
this computer did once, not a promise.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from litenotebook.llm.client import ThinkFilter

TEST_PROMPT = "Reply with exactly one word: ready"
BENCH_PROMPT = (
    "Write three short sentences explaining what a library catalogue is. "
    "Plain text only."
)


@dataclass
class GenerationCheck:
    ok: bool
    models_listed: list[str] = field(default_factory=list)
    reply: str = ""
    ttft_s: float | None = None
    tokens_per_s: float | None = None
    completion_tokens: int | None = None
    seconds: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_generation(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    prompt: str = TEST_PROMPT,
    max_tokens: int = 24,
    timeout_s: float = 180.0,
    transport: httpx.BaseTransport | None = None,
) -> GenerationCheck:
    started = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    result = GenerationCheck(ok=False)
    try:
        with httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_s,
            headers=headers,
            transport=transport,
        ) as client:
            listing = client.get("models")
            listing.raise_for_status()
            data = listing.json().get("data") or []
            result.models_listed = [
                str(m.get("id")) for m in data if isinstance(m, dict)
            ]
            payload = {
                "model": model,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                # the same no-thinking mode Chat uses (Qwen3 via --jinja)
                "chat_template_kwargs": {"enable_thinking": False},
            }
            think = ThinkFilter()
            parts: list[str] = []
            first: float | None = None
            last: float | None = None
            chunks = 0
            timings: dict[str, Any] = {}
            usage: dict[str, Any] = {}
            sent = time.perf_counter()
            with client.stream("POST", "chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = response.read()[:200].decode("utf-8", "replace")
                    result.error = (
                        f"The server returned HTTP {response.status_code}: {body}"
                    )
                    return result
                for line in response.iter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        event = json.loads(body)
                    except ValueError:
                        continue
                    timings = event.get("timings") or timings
                    usage = event.get("usage") or usage
                    for choice in event.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            now = time.perf_counter()
                            first = first or now
                            last = now
                            chunks += 1
                            text = think.feed(delta)
                            if text:
                                parts.append(text)
            parts.append(think.flush())
            result.reply = "".join(parts).strip()
            if first is not None:
                result.ttft_s = round(first - sent, 3)
            result.completion_tokens = usage.get("completion_tokens") or chunks or None
            per_s = timings.get("predicted_per_second")
            if isinstance(per_s, int | float) and per_s > 0:
                result.tokens_per_s = round(float(per_s), 1)
            elif first is not None and last is not None and last > first and chunks > 1:
                result.tokens_per_s = round((chunks - 1) / (last - first), 1)
            result.ok = bool(result.reply)
            if not result.ok:
                result.error = "The model answered with nothing"
    except httpx.ConnectError:
        result.error = "Cannot reach the AI server"
    except httpx.TimeoutException:
        result.error = "The AI server did not answer in time"
    except (httpx.HTTPError, ValueError) as e:
        result.error = f"The AI server gave an invalid answer ({type(e).__name__})"
    finally:
        result.seconds = round(time.perf_counter() - started, 2)
    return result
