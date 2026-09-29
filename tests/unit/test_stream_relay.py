import asyncio

import httpx
import pytest

from gateway.api.streaming import finish_stream, relay_stream
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.queued_request import QueuedRequest


def queued(payload: ChatRequest) -> QueuedRequest:
    return QueuedRequest(
        payload=payload,
        result=asyncio.get_running_loop().create_future(),
        expires_at=0.0,
    )


async def open_stream(client: httpx.AsyncClient) -> httpx.Response:
    return await client.send(client.build_request("POST", "http://worker-a:8000/v1/chat/completions"), stream=True)


def event_engine(request):
    async def events():
        yield b"data: 1\n\n"
        yield b"data: 2\n\n"

    return httpx.Response(200, content=events(), headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_relay_yields_every_chunk_then_closes_upstream_and_releases_dispatch(chat_payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(event_engine)) as client:
        upstream = await open_stream(client)
        pending = queued(chat_payload)

        chunks = [chunk async for chunk in relay_stream(upstream, pending)]

    assert b"".join(chunks) == b"data: 1\n\ndata: 2\n\n"
    assert upstream.is_closed
    assert pending.finished.is_set()


@pytest.mark.asyncio
async def test_client_leaving_early_closes_upstream_and_releases_dispatch(chat_payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(event_engine)) as client:
        upstream = await open_stream(client)
        pending = queued(chat_payload)
        relay = relay_stream(upstream, pending)

        assert await relay.__anext__() == b"data: 1\n\n"
        await relay.aclose()

    assert upstream.is_closed
    assert pending.finished.is_set()


@pytest.mark.asyncio
async def test_finishing_a_stream_that_never_started_still_releases_dispatch(chat_payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(event_engine)) as client:
        upstream = await open_stream(client)
        pending = queued(chat_payload)

        await finish_stream(upstream, pending)
        await finish_stream(upstream, pending)

    assert upstream.is_closed
    assert pending.finished.is_set()
