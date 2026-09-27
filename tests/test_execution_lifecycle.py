import asyncio

import httpx
import pytest

from gateway.execution.lifecycle import cancel_queued_requests, start_worker_tasks
from gateway.execution.queueing import enqueue_request
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.settings import Settings
from gateway.models.workers.worker_state import WorkerState


@pytest.fixture
def setup():
    settings = Settings(
        _env_file=None,
        model_name="test-model",
        worker_urls={"worker-a": "http://worker-a:8000/v1"},
        metrics_interval_s=7,
    )
    worker = WorkerState(id="worker-a", base_url=settings.worker_urls["worker-a"])
    queue = asyncio.Queue(maxsize=8)
    payload = ChatRequest(
        model="test-model", messages=[{"role": "user", "content": "Hello"}]
    )
    return settings, worker, queue, payload


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [1, 2, 3])
async def test_consumers_reach_configured_concurrency_without_exceeding_it(
    setup, monkeypatch, concurrency
):
    settings, worker, queue, payload = setup
    settings.dispatch_concurrency_per_worker = concurrency
    pending = [
        enqueue_request(queue, worker, payload, timeout_s=60)
        for _ in range(concurrency + 1)
    ]
    at_limit = asyncio.Event()
    release = asyncio.Event()
    monitor_stopped = asyncio.Event()
    monitor_calls = []
    active = peak = calls = 0

    async def monitor(client, monitored_worker, interval):
        monitor_calls.append((client, monitored_worker, interval))
        try:
            await asyncio.Event().wait()
        finally:
            monitor_stopped.set()

    async def engine(request):
        nonlocal active, peak, calls
        active += 1
        calls += 1
        peak = max(peak, active)
        if active >= concurrency:
            at_limit.set()
        try:
            await release.wait()
            return httpx.Response(200)
        finally:
            active -= 1

    monkeypatch.setattr("gateway.execution.lifecycle.monitor_worker", monitor)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        async with asyncio.TaskGroup() as group:
            tasks = start_worker_tasks(group, client, worker, queue, settings)
            reached_limit = False
            try:
                try:
                    await asyncio.wait_for(at_limit.wait(), timeout=1)
                    reached_limit = True
                except TimeoutError:
                    pass
                waiting_at_limit = queue.qsize()
                release.set()
                await asyncio.wait_for(queue.join(), timeout=2)
            finally:
                release.set()
                for task in tasks:
                    task.cancel()

    assert reached_limit, f"Expected {concurrency} active requests; observed peak {peak}"
    assert peak == concurrency
    assert waiting_at_limit == 1
    assert calls == concurrency + 1
    assert active == 0
    assert worker.gateway_queue_depth == 0
    assert monitor_calls == [(client, worker, settings.metrics_interval_s)]
    assert monitor_stopped.is_set()
    assert all(task.done() for task in tasks)
    for item in pending:
        assert (await item.result).status_code == 200


@pytest.mark.asyncio
async def test_task_names_distinguish_monitor_and_consumers(setup):
    settings, worker, queue, _ = setup
    settings.dispatch_concurrency_per_worker = 1
    async with httpx.AsyncClient() as client:
        async with asyncio.TaskGroup() as group:
            tasks = start_worker_tasks(group, client, worker, queue, settings)
            names = [task.get_name() for task in tasks]
            for task in tasks:
                task.cancel()

    assert names == ["monitor_worker-a", "dispatch_worker-a_0"]


@pytest.mark.asyncio
async def test_cleanup_cancels_waiters_preserves_finished_results_and_balances_queue(setup):
    _, worker, queue, payload = setup
    waiting, cancelled, completed = [
        enqueue_request(queue, worker, payload, timeout_s=60) for _ in range(3)
    ]
    cancelled.result.cancel()
    response = httpx.Response(200)
    completed.result.set_result(response)

    cancel_queued_requests(queue, worker)
    cancel_queued_requests(queue, worker)
    await asyncio.wait_for(queue.join(), timeout=1)

    assert waiting.result.cancelled()
    assert cancelled.result.cancelled()
    assert completed.result.result() is response
    assert queue.empty()
    assert worker.gateway_queue_depth == 0


@pytest.mark.asyncio
async def test_cleanup_of_empty_queue_resets_depth(setup):
    _, worker, queue, _ = setup
    worker.gateway_queue_depth = 2

    cancel_queued_requests(queue, worker)
    await asyncio.wait_for(queue.join(), timeout=1)

    assert worker.gateway_queue_depth == 0
    assert queue.empty()
