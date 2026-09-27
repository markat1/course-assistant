import asyncio
import importlib
from time import monotonic

import httpx
import pytest

from gateway.execution.dispatch import dispatch_requests
from gateway.execution.queueing import enqueue_request
from gateway.execution.waiting import wait_for_response

waiting_module = importlib.import_module("gateway.execution.waiting")


@pytest.mark.asyncio
async def test_queue_timeout_without_consumer_prevents_later_forwarding(request_queue, worker, chat_payload):
    queue, payload = request_queue, chat_payload
    pending = enqueue_request(queue, worker, payload, timeout_s=60)
    pending.expires_at = monotonic() - 1

    with pytest.raises(TimeoutError, match="Queue deadline exceeded"):
        await asyncio.wait_for(wait_for_response(pending), timeout=1)
    assert pending.started_at is None

    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        task = asyncio.create_task(dispatch_requests(client, worker, queue))
        try:
            await asyncio.wait_for(queue.join(), timeout=1)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert requests == []
    assert worker.gateway_queue_depth == 0


@pytest.mark.asyncio
async def test_queue_deadline_does_not_interrupt_started_upstream_call(request_queue, worker, chat_payload, deadline_timer, monkeypatch):
    queue, payload = request_queue, chat_payload
    pending = enqueue_request(queue, worker, payload, timeout_s=60)
    started = asyncio.Event()
    release = asyncio.Event()

    async def engine(request):
        started.set()
        await release.wait()
        return httpx.Response(200, content=b"finished")

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        consumer = asyncio.create_task(dispatch_requests(client, worker, queue))
        waiter = None
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            assert pending.started_at is not None
            assert pending.started_at < pending.expires_at
            monkeypatch.setattr(waiting_module, "monotonic", lambda: pending.expires_at + 1)
            waiter = asyncio.create_task(wait_for_response(pending))
            await asyncio.wait_for(deadline_timer.fired.wait(), timeout=1)
            assert not waiter.done()
            assert not pending.result.done()
            release.set()
            response = await asyncio.wait_for(waiter, timeout=1)
            assert response.content == b"finished"
            await asyncio.wait_for(queue.join(), timeout=1)
            assert deadline_timer.delays == [0.0]
            assert deadline_timer.handles[0].cancelled()
        finally:
            release.set()
            consumer.cancel()
            if waiter is not None:
                waiter.cancel()
            await asyncio.gather(
                consumer, *([waiter] if waiter is not None else []), return_exceptions=True
            )


@pytest.mark.asyncio
async def test_cancelling_waiter_stops_active_http_call(request_queue, worker, chat_payload, deadline_timer):
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
        consumer = asyncio.create_task(dispatch_requests(client, worker, queue))
        waiter = asyncio.create_task(wait_for_response(pending))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            await asyncio.wait_for(deadline_timer.registered.wait(), timeout=1)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            await asyncio.wait_for(queue.join(), timeout=1)
            assert pending.result.cancelled()
            assert stopped.is_set()
            assert not consumer.done()
            assert deadline_timer.handles[0].cancelled()
        finally:
            consumer.cancel()
            waiter.cancel()
            await asyncio.gather(consumer, waiter, return_exceptions=True)


