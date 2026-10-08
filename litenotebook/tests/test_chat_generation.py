"""Generation internals: true incremental streaming, client disconnect, the guard."""

import json

import anyio
import httpx
import pytest

from conftest import FakeEmbedder, make_pdf
from litenotebook.chat import INTERRUPTED_MESSAGE, get_message, start_exchange
from litenotebook.ingest.pipeline import ingest_pdf
from litenotebook.services.chat import ChatGeneration, GenerationGuard
from litenotebook.services.embedding import SharedEmbedder
from litenotebook.services.settings import SettingsService
from litenotebook.web.streaming import EventStream


@pytest.fixture
def anyio_backend():
    return "asyncio"


def delta(text):
    return (
        f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode()
    )


FINISH = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'


class ClosingStream(httpx.AsyncByteStream):
    """An upstream body that records whether the client closed it."""

    def __init__(self, gen):
        self.gen, self.closed = gen, False

    async def __aiter__(self):
        async for chunk in self.gen:
            yield chunk

    async def aclose(self):
        self.closed = True
        await self.gen.aclose()


@pytest.fixture
def exchange_setup(conn, settings, notebook, store, tmp_path):
    pdf = make_pdf(tmp_path / "z.pdf", ["Quillport is the capital of Zorbania."])
    ingest_pdf(conn, settings, notebook, pdf, FakeEmbedder(), store)

    def make(handler, guard=None):
        guard = guard or GenerationGuard()
        assert guard.try_acquire()
        exchange = start_exchange(conn, notebook.id, None, "Capital of Zorbania?")
        guard.bind(exchange.assistant_message_id)
        return ChatGeneration(
            settings=settings,
            llm=SettingsService(conn, env={}).effective(),
            notebook_id=notebook.id,
            exchange=exchange,
            embedder=SharedEmbedder(FakeEmbedder),
            guard=guard,
            transport=httpx.MockTransport(handler),
        ), guard

    return make


def parse(chunk):
    name, data = chunk.decode().strip().split("\n", 1)
    return name.removeprefix("event: "), json.loads(data.removeprefix("data: "))


@pytest.mark.anyio
async def test_deltas_reach_the_client_before_upstream_finishes(
    exchange_setup, conn, notebook
):
    words = ["Alpha ", "Beta ", "Gamma [S1]"]
    gates = [anyio.Event() for _ in words]

    async def upstream():
        for i, word in enumerate(words):
            if i:
                await gates[i - 1].wait()  # next delta only after the client saw this
            yield delta(word)
        await gates[-1].wait()
        yield FINISH

    async def handler(request):
        return httpx.Response(200, stream=ClosingStream(upstream()))

    generation, guard = exchange_setup(handler)
    received = []
    with anyio.fail_after(10):  # a buffering implementation would deadlock here
        async for chunk in generation.events():
            name, data = parse(chunk)
            if name == "delta":
                received.append(data["t"])
                gates[len(received) - 1].set()
    assert received == ["Alpha ", "Beta ", "Gamma [S1]"]
    msg = get_message(conn, notebook.id, generation.message_id)
    assert msg.status == "complete" and msg.content == "Alpha Beta Gamma [S1]"
    assert guard.active == set()


@pytest.mark.anyio
async def test_client_disconnect_interrupts_and_closes_upstream(
    exchange_setup, conn, notebook
):
    stalled = anyio.Event()

    async def upstream():
        yield delta("Partial answer")
        await anyio.sleep(60)  # llama-server still "generating"
        yield FINISH

    body = ClosingStream(upstream())

    async def handler(request):
        stalled.set()
        return httpx.Response(200, stream=body)

    generation, guard = exchange_setup(handler)
    saw_delta = anyio.Event()
    sent = []
    closed = []

    async def send(message):
        sent.append(message)
        if b"event: delta" in message.get("body", b""):
            saw_delta.set()

    async def receive():
        await saw_delta.wait()
        return {"type": "http.disconnect"}

    response = EventStream(generation.events(), on_close=lambda: closed.append(True))
    with anyio.fail_after(10):
        await response({"type": "http"}, receive, send)

    assert response.disconnected and closed == [True]
    assert body.closed  # upstream llama.cpp response was closed
    msg = get_message(conn, notebook.id, generation.message_id)
    assert msg.status == "interrupted" and msg.error == INTERRUPTED_MESSAGE
    assert msg.content == "Partial answer"
    assert guard.active == set()


@pytest.mark.anyio
async def test_stream_never_iterated_still_releases_and_interrupts(
    exchange_setup, conn, notebook
):
    generation, guard = exchange_setup(lambda r: httpx.Response(200, content=FINISH))

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        raise OSError("client gone before the first byte")

    released = []

    def on_close():
        guard.release(generation.message_id)
        from litenotebook.chat import end_message

        end_message(conn, generation.message_id, "interrupted", INTERRUPTED_MESSAGE)
        released.append(True)

    with anyio.fail_after(10):
        await EventStream(generation.events(), on_close=on_close)(
            {"type": "http"}, receive, send
        )
    assert released == [True] and guard.active == set()
    assert get_message(conn, notebook.id, generation.message_id).status == "interrupted"


def test_guard_allows_one_generation_and_reclaims_stale_leases(monkeypatch):
    from litenotebook.services import chat as chat_service

    guard = GenerationGuard()
    assert guard.try_acquire() and not guard.try_acquire()
    guard.bind(7)
    assert guard.active == {7} and guard.request_stop(7) and guard.stop_requested(7)
    assert not guard.request_stop(8)
    guard.release(7)
    assert guard.active == set() and not guard.stop_requested(7)
    assert guard.try_acquire()
    monkeypatch.setattr(chat_service, "STALE_LEASE_S", 0.0)
    assert guard.try_acquire()  # an abandoned lease does not block forever
