"""Minimal OpenAI-compatible chat client for llama.cpp's llama-server.

Generation only — embeddings never go through this client.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    pass


class LLMResponseError(LLMError):
    """The server was reached but its response was an error or unusable."""


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_s: float
    finish_reason: str | None


def strip_thinking(text: str) -> str:
    """Remove Qwen3 ``<think>…</think>`` blocks, including an unclosed leading one."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():  # opening tag was in the prompt template
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    if text.lstrip().lower().startswith("<think>"):  # truncated mid-thought
        return ""
    return text.strip()


_OPEN_TAG, _CLOSE_TAG = "<think>", "</think>"


def _partial_tag_suffix(text: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that ``text`` ends with."""
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class ThinkFilter:
    """Remove ``<think>…</think>`` from a stream of text deltas.

    Tags may be split across deltas ("<thi" + "nk>"), so a possible partial tag
    at the end of the buffer is held back until the next delta decides it.
    Matching is case-insensitive. Text inside a think block is dropped; an
    unclosed block at the end is dropped too. A stray ``</think>`` (the opening
    tag was in the prompt template) discards whatever came before it if no
    visible text has been emitted yet. Leading whitespace is not emitted.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        self._started = False  # visible non-whitespace text emitted

    def _emit(self, text: str) -> str:
        if not self._started:
            text = text.lstrip()
            if text:
                self._started = True
        return text

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while self._buf:
            low = self._buf.lower()
            if self._inside:
                end = low.find(_CLOSE_TAG)
                if end < 0:
                    keep = _partial_tag_suffix(low, _CLOSE_TAG)
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[end + len(_CLOSE_TAG) :]
                self._inside = False
                continue
            hits = [
                (i, tag) for tag in (_OPEN_TAG, _CLOSE_TAG) if (i := low.find(tag)) >= 0
            ]
            if hits:
                index, tag = min(hits)
                before = self._buf[:index]
                self._buf = self._buf[index + len(tag) :]
                if tag == _OPEN_TAG:
                    out.append(before)
                    self._inside = True
                elif self._started or "".join(out).strip():
                    out.append(before)  # stray close tag after real text: drop tag
                else:
                    out.clear()  # everything so far was un-tagged thinking
                continue
            keep = max(_partial_tag_suffix(low, tag) for tag in (_OPEN_TAG, _CLOSE_TAG))
            out.append(self._buf[: len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep :]
            break
        return self._emit("".join(out))

    def flush(self) -> str:
        """End of stream: an unfinished tag prefix is literal text."""
        rest, self._buf = ("" if self._inside else self._buf), ""
        return self._emit(rest)


@dataclass
class StreamInfo:
    finish_reason: str | None = None
    model: str | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    done: bool = False


class ChatStream:
    """Streaming chat completion from an OpenAI-compatible server (async).

    ``deltas()`` yields raw content deltas as llama-server produces them.
    Leaving the iteration early (cancellation, client disconnect) closes the
    upstream HTTP response, which makes llama-server stop generating.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        disable_thinking: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.disable_thinking = disable_thinking
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)
        self._transport = transport
        self.info = StreamInfo()

    async def deltas(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if self.model:
            payload["model"] = self.model
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                async with client.stream(
                    "POST", "chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread())[:300]
                        raise LLMResponseError(
                            f"LLM server returned HTTP {response.status_code}: "
                            f"{body.decode('utf-8', 'replace')}"
                        )
                    async for line in response.aiter_lines():
                        for delta in self._parse_line(line):
                            yield delta
                        if self.info.done:
                            return
            if self.info.finish_reason is None:
                raise LLMResponseError("The LLM stream ended unexpectedly")
        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot reach the LLM server at {self.base_url} — is "
                f"llama-server running?"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError("The LLM server timed out") from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {type(e).__name__}") from e

    def _parse_line(self, line: str) -> list[str]:
        line = line.strip()
        if not line or line.startswith(":"):
            return []  # keep-alive / SSE comment
        field_name, _, value = line.partition(":")
        if field_name in ("event", "id", "retry"):
            return []
        if field_name != "data":
            raise LLMResponseError("The LLM server sent malformed streaming data")
        value = value.strip()
        if value == "[DONE]":
            self.info.done = True
            return []
        try:
            obj = json.loads(value)
        except ValueError as e:
            raise LLMResponseError(
                "The LLM server sent malformed streaming data"
            ) from e
        if not isinstance(obj, dict):
            raise LLMResponseError("The LLM server sent malformed streaming data")
        if "error" in obj:
            err = obj["error"]
            message = err.get("message") if isinstance(err, dict) else err
            raise LLMResponseError(f"LLM server error: {str(message)[:300]}")
        if isinstance(obj.get("model"), str):
            self.info.model = obj["model"]
        if isinstance(obj.get("timings"), dict):
            self.info.timings = obj["timings"]
        choices = obj.get("choices")
        if not isinstance(choices, list):
            raise LLMResponseError("The LLM server sent malformed streaming data")
        out: list[str] = []
        for choice in choices[:1]:
            if not isinstance(choice, dict):
                raise LLMResponseError("The LLM server sent malformed streaming data")
            if choice.get("finish_reason"):
                self.info.finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                out.append(content)  # reasoning_content is deliberately ignored
        return out


class StreamStopped(LLMError):
    """``should_stop`` returned True; the upstream response was closed."""


def complete_streaming(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    timeout_s: float = 180.0,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    extra: dict[str, Any] | None = None,
    transport: httpx.BaseTransport | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[str, StreamInfo]:
    """A whole completion, streamed (sync), for background work such as quiz
    generation. Streaming keeps it cancellable: when ``should_stop`` returns
    True the response is closed, which makes llama-server stop generating.
    Thinking is disabled and any ``<think>`` text is removed."""
    parser = ChatStream(base_url, model, api_key=api_key)
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        **(extra or {}),
    }
    if model:
        payload["model"] = model
    parts: list[str] = []
    think = ThinkFilter()
    try:
        with httpx.Client(
            base_url=parser.base_url,
            headers=parser._headers,
            timeout=parser._timeout,
            transport=transport,
        ) as client:
            with client.stream("POST", "chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = response.read()[:300]
                    raise LLMResponseError(
                        f"LLM server returned HTTP {response.status_code}: "
                        f"{body.decode('utf-8', 'replace')}"
                    )
                for line in response.iter_lines():
                    for delta in parser._parse_line(line):
                        parts.append(think.feed(delta))
                    if should_stop is not None and should_stop():
                        raise StreamStopped("Stopped")
                    if parser.info.done:
                        break
        if parser.info.finish_reason is None:
            raise LLMResponseError("The LLM stream ended unexpectedly")
    except httpx.ConnectError as e:
        raise LLMError(
            f"Cannot reach the LLM server at {parser.base_url} — is llama-server "
            "running?"
        ) from e
    except httpx.TimeoutException as e:
        raise LLMError("The LLM server timed out") from e
    except httpx.HTTPError as e:
        raise LLMError(f"LLM request failed: {type(e).__name__}") from e
    parts.append(think.flush())
    return "".join(parts), parser.info


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        disable_thinking: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.disable_thinking = disable_thinking
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot reach the LLM server at {self._http.base_url} — is "
                f"llama-server running?"
            ) from e
        except httpx.TimeoutException as e:
            raise LLMError("The LLM server timed out") from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e
        if response.status_code >= 400:
            raise LLMResponseError(
                f"LLM server returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as e:
            raise LLMResponseError("LLM server returned invalid JSON") from e

    def list_models(self) -> list[str]:
        data = self._request("GET", "models")
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise LLMResponseError(
                "The server's /models response is not OpenAI-compatible — "
                "check that the endpoint URL ends in /v1"
            )
        return [
            m["id"]
            for m in items
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        ]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self.disable_thinking:
            # Qwen3 no-thinking mode via llama-server's --jinja chat template.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        started = time.perf_counter()
        data = self._request("POST", "chat/completions", json=payload)
        elapsed = time.perf_counter() - started
        try:
            choice = data["choices"][0]
            raw = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("Unexpected response shape from LLM server") from e
        usage = data.get("usage") or {}
        return ChatResult(
            content=strip_thinking(raw),
            model=data.get("model"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            elapsed_s=elapsed,
            finish_reason=choice.get("finish_reason"),
        )
