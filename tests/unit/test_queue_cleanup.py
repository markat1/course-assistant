import asyncio

import httpx
import pytest
import pytest_asyncio

from gateway.execution.lifecycle import cancel_queued_requests
from gateway.execution.queueing import enqueue_request


@pytest_asyncio.fixture
async def pending(request_queue, worker, chat_payload):
    request = enqueue_request(request_queue, worker, chat_payload, timeout_s=60)
    try:
        yield request
    finally:
        cancel_queued_requests(request_queue, worker)


@pytest.mark.asyncio
async def test_cleanup_cancels_waiting_work_and_releases_queue(request_queue, worker, pending):
    cancel_queued_requests(request_queue, worker)

    assert pending.result.cancelled()
    assert request_queue.empty()
    assert worker.gateway_queue_depth == 0
    await asyncio.wait_for(request_queue.join(), timeout=1)


@pytest.mark.asyncio
async def test_cleanup_preserves_a_finished_result(request_queue, worker, pending):
    response = httpx.Response(200)
    pending.result.set_result(response)

    cancel_queued_requests(request_queue, worker)

    assert pending.result.result() is response
    await asyncio.wait_for(request_queue.join(), timeout=1)


@pytest.mark.asyncio
async def test_cleanup_handles_already_cancelled_work(request_queue, worker, pending):
    pending.result.cancel()

    cancel_queued_requests(request_queue, worker)

    assert request_queue.empty()
    assert pending.result.cancelled()
    await asyncio.wait_for(request_queue.join(), timeout=1)


@pytest.mark.asyncio
async def test_repeated_cleanup_does_not_unbalance_queue(request_queue, worker, pending):
    cancel_queued_requests(request_queue, worker)

    cancel_queued_requests(request_queue, worker)

    assert pending.result.cancelled()
    await asyncio.wait_for(request_queue.join(), timeout=1)


@pytest.mark.asyncio
async def test_empty_queue_cleanup_clears_outdated_depth(request_queue, worker):
    worker.gateway_queue_depth = 2

    cancel_queued_requests(request_queue, worker)

    assert worker.gateway_queue_depth == 0
    await asyncio.wait_for(request_queue.join(), timeout=1)
