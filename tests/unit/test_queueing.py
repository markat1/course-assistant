import asyncio
import pytest

from gateway.execution.queueing import enqueue_request

@pytest.fixture
def request_queue():
    return asyncio.Queue(maxsize=1)

@pytest.mark.asyncio
async def test_enqueue_tracks_depth_and_deadline(request_queue, worker, chat_payload, monkeypatch):
    queue, payload = request_queue, chat_payload
    monkeypatch.setattr("gateway.execution.queueing.monotonic", lambda: 100.0)

    pending = enqueue_request(queue, worker, payload, timeout_s=5)

    assert worker.gateway_queue_depth == 1
    assert pending.expires_at == 105.0
    assert pending.payload == payload
    assert not pending.result.done()
    assert queue.get_nowait() is pending

    queue.task_done()
    pending.result.cancel()

@pytest.mark.asyncio
async def test_full_queue_preserves_existing_request(request_queue, worker, chat_payload):
    queue, payload = request_queue, chat_payload
    first = enqueue_request(queue, worker, payload, timeout_s=5)

    with pytest.raises(asyncio.QueueFull):
        enqueue_request(queue, worker, payload, timeout_s=5)

    assert queue.qsize() == 1
    assert worker.gateway_queue_depth == 1
    assert queue.get_nowait() is first

    queue.task_done()
    first.result.cancel()
