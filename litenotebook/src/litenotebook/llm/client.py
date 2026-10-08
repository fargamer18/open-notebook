"""Minimal OpenAI-compatible chat client for llama.cpp's llama-server.

Generation only — embeddings never go through this client.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
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
