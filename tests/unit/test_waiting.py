import asyncio

import httpx
import pytest
import pytest_asyncio

from gateway.execution.waiting import expire_queued_request, wait_for_response
from gateway.models.queued_request import QueuedRequest


@pytest_asyncio.fixture
async def pending(chat_payload, monkeypatch):
    monkeypatch.setattr("gateway.execution.waiting.monotonic", lambda: 100.0)
    return QueuedRequest(
        payload=chat_payload,
        result=asyncio.get_running_loop().create_future(),
        expires_at=105.0,
    )


@pytest_asyncio.fixture
async def waiter(pending, deadline_timer):
    task = asyncio.create_task(wait_for_response(pending))
    try:
        await asyncio.wait_for(deadline_timer.registered.wait(), timeout=1)
        yield task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_waiting_returns_worker_response_and_releases_timer(pending, waiter, deadline_timer):
    response = httpx.Response(503)

    pending.result.set_result(response)

    assert await waiter is response
    assert deadline_timer.handles[0].cancelled()


@pytest.mark.asyncio
async def test_waiting_preserves_transport_error_and_releases_timer(pending, waiter, deadline_timer):
    error = httpx.ReadTimeout("worker timed out")

    pending.result.set_exception(error)

    with pytest.raises(httpx.ReadTimeout) as caught:
        await waiter
    assert caught.value is error
    assert deadline_timer.handles[0].cancelled()


@pytest.mark.asyncio
async def test_cancelling_waiter_cancels_request_and_releases_timer(pending, waiter, deadline_timer):
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert pending.result.cancelled()
    assert deadline_timer.handles[0].cancelled()


@pytest.mark.asyncio
async def test_waiting_schedules_only_the_remaining_queue_time(waiter, deadline_timer):
    assert deadline_timer.delays == [5.0]


@pytest.mark.asyncio
async def test_expiry_rejects_work_that_has_not_started(pending):
    expire_queued_request(pending)

    with pytest.raises(TimeoutError, match="Queue deadline exceeded"):
        await pending.result


@pytest.mark.asyncio
async def test_expiry_leaves_started_work_running(pending):
    pending.started_at = 104.0

    expire_queued_request(pending)

    assert not pending.result.done()
    pending.result.cancel()


@pytest.mark.asyncio
async def test_expiry_preserves_a_completed_response(pending):
    response = httpx.Response(200)
    pending.result.set_result(response)

    expire_queued_request(pending)

    assert pending.result.result() is response


@pytest.mark.asyncio
async def test_expiry_preserves_a_transport_error(pending):
    error = httpx.ConnectError("worker unavailable")
    pending.result.set_exception(error)

    expire_queued_request(pending)

    assert pending.result.exception() is error


@pytest.mark.asyncio
async def test_expiry_preserves_cancellation(pending):
    pending.result.cancel()

    expire_queued_request(pending)

    assert pending.result.cancelled()
