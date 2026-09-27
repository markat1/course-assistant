import asyncio
import importlib
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from gateway.execution.lifecycle import manage_worker_tasks
from gateway.execution.queueing import enqueue_request
from gateway.lifespan import create_queues, create_workers
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.settings import Settings


lifespan_module = importlib.import_module("gateway.lifespan")
task_module = importlib.import_module("gateway.execution.lifecycle")


def make_settings(worker_count):
    return Settings(
        _env_file=None,
        model_name="test-model",
        worker_urls={
            f"worker-{index}": f"http://worker-{index}:8000/v1"
            for index in range(worker_count)
        },
        dispatch_concurrency_per_worker=1,
        queue_max_size=4,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_count", [1, 2])
async def test_lifespan_starts_all_workers_and_cleans_up_before_closing_client(
    monkeypatch, worker_count
):
    settings = make_settings(worker_count)
    started = {worker_id: asyncio.Event() for worker_id in settings.worker_urls}
    events = []
    logging_levels = []
    client_open_during_cleanup = []
    pending = []
    payload = ChatRequest(
        model="test-model", messages=[{"role": "user", "content": "Hello"}]
    )

    async def engine(request):
        worker_id = request.url.host
        events.append(("engine_started", worker_id))
        started[worker_id].set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append(("engine_stopped", worker_id))
            client_open_during_cleanup.append(not client.is_closed)

    async def monitor(client, worker, interval):
        assert logging_levels == ["INFO"]
        events.append(("monitor_started", worker.id))
        try:
            await asyncio.Event().wait()
        finally:
            events.append(("monitor_stopped", worker.id))

    real_cleanup = task_module.cancel_queued_requests

    def cleanup(queue, worker):
        events.append(("queue_cleanup", worker.id))
        client_open_during_cleanup.append(not client.is_closed)
        real_cleanup(queue, worker)

    client = httpx.AsyncClient(transport=httpx.MockTransport(engine))
    monkeypatch.setattr(lifespan_module, "Settings", lambda: settings)
    monkeypatch.setattr(
        lifespan_module, "configure_logging", logging_levels.append, raising=False
    )
    monkeypatch.setattr(
        lifespan_module, "httpx", SimpleNamespace(AsyncClient=lambda **kwargs: client)
    )
    monkeypatch.setattr(task_module, "monitor_worker", monitor)
    monkeypatch.setattr(task_module, "cancel_queued_requests", cleanup)
    app = FastAPI()

    async with lifespan_module.lifespan(app):
        assert app.state.client is client
        for worker_id, worker in app.state.workers.items():
            for _ in range(2):
                pending.append(
                    enqueue_request(
                        app.state.queues[worker_id], worker, payload, timeout_s=60
                    )
                )
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started.values())), timeout=1
        )
        assert all(queue.qsize() == 1 for queue in app.state.queues.values())

    assert client.is_closed
    assert all(client_open_during_cleanup)
    assert all(item.result.cancelled() for item in pending)
    for worker_id, queue in app.state.queues.items():
        await asyncio.wait_for(queue.join(), timeout=1)
        assert queue.empty()
        assert app.state.workers[worker_id].gateway_queue_depth == 0
        assert events.index(("engine_stopped", worker_id)) < events.index(
            ("queue_cleanup", worker_id)
        )
        assert events.index(("monitor_stopped", worker_id)) < events.index(
            ("queue_cleanup", worker_id)
        )
        assert events.count(("engine_started", worker_id)) == 1


@pytest.mark.asyncio
async def test_task_context_enters_and_exits_once_with_two_workers():
    settings = make_settings(2)
    workers = create_workers(settings)
    queues = create_queues(workers, settings.queue_max_size)
    async with httpx.AsyncClient() as client:
        async with manage_worker_tasks(client, workers, queues, settings):
            pass


@pytest.mark.asyncio
async def test_monitor_failure_propagates_after_cancelling_active_and_queued_work(monkeypatch):
    settings = make_settings(1)
    workers = create_workers(settings)
    queues = create_queues(workers, settings.queue_max_size)
    worker = workers["worker-0"]
    queue = queues[worker.id]
    started = asyncio.Event()
    stopped = asyncio.Event()
    payload = ChatRequest(
        model="test-model", messages=[{"role": "user", "content": "Hello"}]
    )
    pending = [enqueue_request(queue, worker, payload, timeout_s=60) for _ in range(2)]

    async def engine(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def broken_monitor(client, worker, interval):
        await started.wait()
        raise RuntimeError("monitor crashed")

    monkeypatch.setattr(task_module, "monitor_worker", broken_monitor)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(ExceptionGroup) as caught:
            async with asyncio.timeout(2):
                async with manage_worker_tasks(client, workers, queues, settings):
                    await asyncio.Event().wait()
        assert not client.is_closed

    assert len(caught.value.exceptions) == 1
    assert isinstance(caught.value.exceptions[0], RuntimeError)
    assert str(caught.value.exceptions[0]) == "monitor crashed"
    assert stopped.is_set()
    assert all(item.result.cancelled() for item in pending)
    await asyncio.wait_for(queue.join(), timeout=1)
    assert worker.gateway_queue_depth == 0
