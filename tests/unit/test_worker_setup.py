import asyncio

import pytest

from gateway.lifespan import create_queues, create_workers
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.settings import Settings
from gateway.execution.queueing import enqueue_request


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        model_name="test-model",
        worker_urls={
            "worker-a": "http://worker-a:8000/v1",
            "worker-b": "http://worker-b:8000/v1",
        },
        queue_max_size=1,
    )


def test_configured_workers_start_unready_with_unknown_metrics(settings):
    workers = create_workers(settings)

    assert set(workers) == set(settings.worker_urls)
    for worker_id, worker in workers.items():
        assert worker.id == worker_id
        assert worker.base_url == settings.worker_urls[worker_id]
        assert worker.ready is False
        assert worker.metrics_updated_at is None
        assert worker.engine_running is None
        assert worker.engine_waiting is None
        assert worker.kv_cache_usage_ratio is None


@pytest.mark.asyncio
async def test_full_worker_queue_does_not_consume_another_workers_capacity(settings):
    workers = create_workers(settings)
    queues = create_queues(workers, settings.queue_max_size)
    payload = ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert set(queues) == set(workers)
    first = enqueue_request(queues["worker-a"], workers["worker-a"], payload, timeout_s=5)
    try:
        with pytest.raises(asyncio.QueueFull):
            enqueue_request(queues["worker-a"], workers["worker-a"], payload, timeout_s=5)

        second = enqueue_request(queues["worker-b"], workers["worker-b"], payload, timeout_s=5)
        try:
            assert queues["worker-a"].get_nowait() is first
            queues["worker-a"].task_done()
            assert queues["worker-b"].get_nowait() is second
            queues["worker-b"].task_done()
        finally:
            second.result.cancel()
    finally:
        first.result.cancel()
