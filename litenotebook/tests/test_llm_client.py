import json

import httpx
import pytest

from litenotebook.llm.client import LLMClient, LLMError, strip_thinking


def completion(content, **extra):
    return {
        "model": "qwen",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 7},
        **extra,
    }


def client_with(handler):
    return LLMClient(
        "http://llm.test/v1", "qwen3-4b", transport=httpx.MockTransport(handler)
    )


def test_chat_payload_and_parsing():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion("Answer [S1]."))

    with client_with(handler) as llm:
        result = llm.chat([{"role": "user", "content": "hi"}], max_tokens=64)
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    body = seen["body"]
    assert body["model"] == "qwen3-4b" and body["max_tokens"] == 64
    assert body["stream"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert result.content == "Answer [S1]."
    assert (result.prompt_tokens, result.completion_tokens) == (50, 7)
    assert result.finish_reason == "stop"


def test_thinking_is_stripped_from_response():
    handler = lambda r: httpx.Response(  # noqa: E731
        200, json=completion("<think>\nplanning...\n</think>\n\nFinal [S2].")
    )
    with client_with(handler) as llm:
        assert llm.chat([]).content == "Final [S2]."


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("<think>a</think>Answer", "Answer"),
        ("<THINK>a\nb</THINK>  Answer  ", "Answer"),
        ("reasoning without opening tag</think>Answer", "Answer"),
        ("<think>never closed because max_tokens ran out", ""),
        ("Plain answer", "Plain answer"),
    ],
)
def test_strip_thinking(raw, clean):
    assert strip_thinking(raw) == clean


def test_list_models():
    handler = lambda r: httpx.Response(200, json={"data": [{"id": "qwen3"}]})  # noqa: E731
    with client_with(handler) as llm:
        assert llm.list_models() == ["qwen3"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"choices": []}),
    ],
)
def test_server_errors_raise_llm_error(response):
    with client_with(lambda r: response) as llm, pytest.raises(LLMError):
        llm.chat([])


def test_connection_refused_is_friendly():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with client_with(handler) as llm, pytest.raises(LLMError, match="llama-server"):
        llm.chat([])
