import asyncio
import json

import httpx
import pytest

from gateway.execution.dispatch import dispatch_one, dispatch_requests
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.queued_request import QueuedRequest
from gateway.models.workers.worker_state import WorkerState
from gateway.execution.queueing import enqueue_request


@pytest.fixture
def dispatch_setup():
    queue = asyncio.Queue[QueuedRequest](maxsize=4)
    worker = WorkerState(id="worker-a", base_url="http://worker-a:8000/v1")
    payload = ChatRequest(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "Explain admission control."}],
        max_tokens=32,
    )
    return queue, worker, payload


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
async def test_response_is_preserved_without_retry(dispatch_setup, status):
    queue, worker, payload = dispatch_setup
    pending = enqueue_request(queue, worker, payload, timeout_s=60)
    requests = []
    body = b'{"message":"engine response"}'

    def engine(request):
        requests.append(request)
        assert worker.gateway_queue_depth == 0
        return httpx.Response(status, content=body)

    await drain(queue, worker, engine)
    response = await pending.result

    assert response.status_code == status
    assert response.content == body
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://worker-a:8000/v1/chat/completions"
    assert json.loads(requests[0].content) == payload.model_dump(
        mode="json", exclude_unset=True
    )


@pytest.mark.asyncio
async def test_expired_request_never_reaches_engine(dispatch_setup, monkeypatch):
    queue, worker, payload = dispatch_setup
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
async def test_cancelled_queued_request_never_reaches_engine(dispatch_setup):
    queue, worker, payload = dispatch_setup
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
async def test_transport_error_does_not_stop_next_request(dispatch_setup, error_type):
    queue, worker, payload = dispatch_setup
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
async def test_shutdown_cancels_active_request_and_finishes_queue_item(dispatch_setup):
    queue, worker, payload = dispatch_setup
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
@pytest.mark.parametrize("now, expected_calls", [(104.999, 1), (105.0, 0)])
async def test_single_dispatch_checks_deadline_without_queue_loop(
    dispatch_setup, monkeypatch, now, expected_calls
):
    _, worker, payload = dispatch_setup
    pending = QueuedRequest(
        payload=payload,
        result=asyncio.get_running_loop().create_future(),
        expires_at=105.0,
    )
    monkeypatch.setattr("gateway.execution.dispatch.monotonic", lambda: now)
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        await dispatch_one(client, worker, pending)

    if expected_calls:
        assert (await pending.result).status_code == 200
    else:
        with pytest.raises(TimeoutError, match="Queue deadline exceeded"):
            await pending.result
    assert len(requests) == expected_calls


@pytest.mark.asyncio
async def test_caller_cancellation_stops_upstream_and_allows_next_request(dispatch_setup):
    queue, worker, payload = dispatch_setup
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
