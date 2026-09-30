import asyncio
from time import monotonic
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from gateway.execution.dispatch import dispatch_requests
from gateway.execution.lifecycle import cancel_queued_requests
from gateway.lifespan import create_queues
from gateway.main import app
from gateway.models.workers.worker_state import WorkerState
from gateway.policies.tenant_window import TenantWindow

CHAT_URL = "/v1/chat/completions"


def payload(question):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 16,
    }


@pytest_asyncio.fixture
async def gateway(monkeypatch):
    """One ready worker whose queue is filled before its dispatcher starts."""
    worker = WorkerState(
        id="worker-a", base_url="http://worker-a:8000/v1", ready=True,
        engine_running=0, engine_waiting=0, kv_cache_usage_ratio=0.1,
        metrics_updated_at=monotonic(),
    )
    queue = create_queues({worker.id: worker}, max_size=8)[worker.id]
    dispatched = []

    def engine(request):
        dispatched.append(request.read().decode())
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(app.state, "workers", {worker.id: worker}, raising=False)
    monkeypatch.setattr(app.state, "queues", {worker.id: queue}, raising=False)
    monkeypatch.setattr(
        app.state, "settings",
        SimpleNamespace(
            metrics_max_age_s=10.0, kv_usage_limit=0.9, queue_timeout_s=5,
            capacity_retry_after_s=7, queue_max_size=8,
            context_length=8192, max_output_tokens=1024,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        app.state, "tenant_window", TenantWindow(max_tokens=10**9, window_s=60.0), raising=False,
    )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(engine)) as upstream,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client,
    ):
        tasks = []

        def start_dispatcher():
            tasks.append(asyncio.create_task(dispatch_requests(upstream, worker, queue)))

        async def wait_until_queued(count):
            async with asyncio.timeout(1):
                while queue.qsize() < count:
                    await asyncio.sleep(0)

        try:
            yield SimpleNamespace(
                client=client, dispatched=dispatched,
                start_dispatcher=start_dispatcher, wait_until_queued=wait_until_queued,
            )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            cancel_queued_requests(queue, worker)


@pytest.mark.asyncio
async def test_queued_interactive_turn_is_dispatched_before_earlier_batch(gateway):
    batch = asyncio.create_task(gateway.client.post(
        CHAT_URL, json=payload("batch sweep"), headers={"X-Request-Class": "batch"},
    ))
    await gateway.wait_until_queued(1)
    interactive = asyncio.create_task(gateway.client.post(
        CHAT_URL, json=payload("student turn"), headers={"X-Request-Class": "interactive"},
    ))
    await gateway.wait_until_queued(2)

    gateway.start_dispatcher()
    responses = await asyncio.gather(batch, interactive)

    assert [response.status_code for response in responses] == [200, 200]
    assert "student turn" in gateway.dispatched[0]
    assert "batch sweep" in gateway.dispatched[1]


@pytest.mark.asyncio
async def test_request_without_a_class_is_served_as_interactive(gateway):
    batch = asyncio.create_task(gateway.client.post(
        CHAT_URL, json=payload("batch sweep"), headers={"X-Request-Class": "batch"},
    ))
    await gateway.wait_until_queued(1)
    unlabelled = asyncio.create_task(gateway.client.post(CHAT_URL, json=payload("app turn")))
    await gateway.wait_until_queued(2)

    gateway.start_dispatcher()
    await asyncio.gather(batch, unlabelled)

    assert "app turn" in gateway.dispatched[0]


@pytest.mark.asyncio
async def test_unknown_request_class_is_rejected_by_the_guard(gateway):
    response = await gateway.client.post(
        CHAT_URL, json=payload("hello"), headers={"X-Request-Class": "urgent"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad_request_class"
    assert gateway.dispatched == []
