import asyncio

import httpx
import pytest
import pytest_asyncio

from gateway.execution.dispatch import dispatch_one
from gateway.models.queued_request import QueuedRequest


@pytest_asyncio.fixture
async def pending(chat_payload):
    return QueuedRequest(
        payload=chat_payload,
        result=asyncio.get_running_loop().create_future(),
        expires_at=105.0,
    )


@pytest.mark.asyncio
async def test_dispatch_starts_just_before_queue_deadline(pending, worker, monkeypatch):
    monkeypatch.setattr("gateway.execution.dispatch.monotonic", lambda: 104.999)
    response = httpx.Response(200)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: response)
    ) as client:
        await dispatch_one(client, worker, pending)

    assert await pending.result is response
    assert pending.started_at == 104.999


@pytest.mark.asyncio
async def test_dispatch_rejects_at_queue_deadline_without_calling_worker(pending, worker, monkeypatch):
    monkeypatch.setattr("gateway.execution.dispatch.monotonic", lambda: 105.0)
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        await dispatch_one(client, worker, pending)

    with pytest.raises(TimeoutError, match="Queue deadline exceeded"):
        await pending.result
    assert requests == []
    assert pending.started_at is None
