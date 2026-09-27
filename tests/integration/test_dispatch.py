import asyncio

import httpx
import pytest

from gateway.execution.dispatch import dispatch_requests
from gateway.execution.queueing import enqueue_request


async def drain(queue, worker, handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(dispatch_requests(client, worker, queue))
        try:
            await asyncio.wait_for(queue.join(), timeout=2)
            assert not task.done(), "Dispatcher stopped instead of waiting for more work"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 429, 500, 503, 529])
async def test_response_is_preserved_without_retry(request_queue, worker, chat_payload, status):
    queue, payload = request_queue, chat_payload
    pending = enqueue_request(queue, worker, payload, timeout_s=60)
    requests = []
    body = b'{"message":"engine response"}'

    def engine(request):
        requests.append(request)
        return httpx.Response(status, content=body)

    await drain(queue, worker, engine)
    response = await pending.result

    assert response.status_code == status
    assert response.content == body
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_expired_request_never_reaches_engine(request_queue, worker, chat_payload, monkeypatch):
    queue, payload = request_queue, chat_payload
    monkeypatch.setattr("gateway.execution.queueing.monotonic", lambda: 100.0)
    monkeypatch.setattr("gateway.execution.dispatch.monotonic", lambda: 105.0)
    pending = enqueue_request(queue, worker, payload, timeout_s=5)
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200)

    await drain(queue, worker, engine)

    with pytest.raises(TimeoutError, match="Queue deadline exceeded"):
        await pending.result
    assert requests == []
    assert worker.gateway_queue_depth == 0


@pytest.mark.asyncio
async def test_cancelled_queued_request_never_reaches_engine(request_queue, worker, chat_payload):
    queue, payload = request_queue, chat_payload
    pending = enqueue_request(queue, worker, payload, timeout_s=60)
    pending.result.cancel()
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200)

    await drain(queue, worker, engine)

    assert pending.result.cancelled()
    assert requests == []
    assert worker.gateway_queue_depth == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_error_does_not_stop_next_request(request_queue, worker, chat_payload, error_type):
    queue, payload = request_queue, chat_payload
    first = enqueue_request(queue, worker, payload, timeout_s=60)
    second = enqueue_request(queue, worker, payload, timeout_s=60)
    requests = []

    def engine(request):
        requests.append(request)
        if len(requests) == 1:
            raise error_type("Engine unavailable", request=request)
        return httpx.Response(200, json={"choices": []})

    await drain(queue, worker, engine)

    with pytest.raises(error_type, match="Engine unavailable"):
        await first.result
    assert (await second.result).status_code == 200
    assert len(requests) == 2
    assert worker.gateway_queue_depth == 0


@pytest.mark.asyncio
async def test_shutdown_cancels_active_request_and_finishes_queue_item(request_queue, worker, chat_payload):
    queue, payload = request_queue, chat_payload
    pending = enqueue_request(queue, worker, payload, timeout_s=60)
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def engine(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        task = asyncio.create_task(dispatch_requests(client, worker, queue))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(queue.join(), timeout=2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert stopped.is_set()
    assert pending.result.cancelled()
    assert worker.gateway_queue_depth == 0


@pytest.mark.asyncio
async def test_caller_cancellation_stops_upstream_and_allows_next_request(request_queue, worker, chat_payload):
    queue, payload = request_queue, chat_payload
    first = enqueue_request(queue, worker, payload, timeout_s=60)
    second = enqueue_request(queue, worker, payload, timeout_s=60)
    started = asyncio.Event()
    stopped = asyncio.Event()
    requests = []

    async def engine(request):
        requests.append(request)
        if len(requests) == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        assert stopped.is_set()
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        task = asyncio.create_task(dispatch_requests(client, worker, queue))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            first.result.cancel()
            await asyncio.wait_for(queue.join(), timeout=2)
            assert not task.done()
            assert first.result.cancelled()
            assert (await second.result).status_code == 200
            assert stopped.is_set()
            assert len(requests) == 2
            assert worker.gateway_queue_depth == 0
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
