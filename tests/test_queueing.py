import asyncio
import pytest

from gateway.models.chat_request import ChatRequest
from gateway.models.queued_request import QueuedRequest
from gateway.models.worker_state import WorkerState
from gateway.queueing import enqueue_request

@pytest.fixture
def queue_setup():
    queue = asyncio.Queue[QueuedRequest](maxsize=1)
    worker = WorkerState(
        id="worker-a",
        base_url="http://worker-a:8000/v1",
    )
    payload = ChatRequest(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "Explain admission control."}]
    )
    return queue, worker, payload

@pytest.mark.asyncio
async def test_enqueue_tracks_depth_and_deadline(queue_setup, monkeypatch):
    queue, worker, payload = queue_setup
    monkeypatch.setattr("gateway.queueing.monotonic", lambda: 100.0)

    pending = enqueue_request(queue, worker, payload, timeout_s=5)

    assert worker.gateway_queue_depth == 1
    assert pending.expires_at == 105.0
    assert pending.payload == payload
    assert not pending.result.done()
    assert queue.get_nowait() is pending

    queue.task_done()
    pending.result.cancel()

@pytest.mark.asyncio
async def test_full_queue_preserves_existing_request(queue_setup):
    queue, worker, payload = queue_setup
    first = enqueue_request(queue, worker, payload, timeout_s=5)

    with pytest.raises(asyncio.QueueFull):
        enqueue_request(queue, worker, payload, timeout_s=5)

    assert queue.qsize() == 1
    assert worker.gateway_queue_depth == 1
    assert queue.get_nowait() is first

    queue.task_done()
    first.result.cancel()
