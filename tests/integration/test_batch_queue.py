import asyncio
from time import monotonic
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi.responses import JSONResponse

from gateway.execution.lifecycle import cancel_queued_requests
from gateway.lifespan import create_queues
from gateway.main import app
from gateway.models.workers.worker_state import WorkerState
from gateway.monitoring.metrics import GATEWAY_REGISTRY
from gateway.policies.tenant_window import TenantWindow

CHAT_URL = "/v1/chat/completions"
BATCH = {"X-Request-Class": "batch"}
INTERACTIVE = {"X-Request-Class": "interactive"}


def payload(question):
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 16,
    }


def queue_full_sheds():
    labels = {"reason": "queue_full", "code": "503"}
    return GATEWAY_REGISTRY.get_sample_value("orch_shed_total", labels) or 0.0


@pytest_asyncio.fixture
async def gateway(monkeypatch):
    """One ready worker with a queue of 4 that nothing dispatches; batch may hold 2 of them."""
    worker = WorkerState(
        id="worker-a", base_url="http://worker-a:8000/v1", ready=True,
        engine_running=0, engine_waiting=0, kv_cache_usage_ratio=0.1,
        metrics_updated_at=monotonic(),
    )
    queue = create_queues({worker.id: worker}, max_size=4)[worker.id]

    monkeypatch.setattr(app.state, "workers", {worker.id: worker}, raising=False)
    monkeypatch.setattr(app.state, "queues", {worker.id: queue}, raising=False)
    monkeypatch.setattr(
        app.state, "settings",
        SimpleNamespace(
            metrics_max_age_s=10.0, kv_usage_limit=0.9, capacity_retry_after_s=7,
            queue_max_size=4, queue_timeout_s=5,
            batch_queue_timeout_s=30, batch_queue_max_size=2,
            context_length=8192, max_output_tokens=1024, prefix_load_slack=4, dispatch_concurrency_per_worker=8,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        app.state, "tenant_window", TenantWindow(max_tokens=10**9, window_s=60.0), raising=False,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as client:
        waiting = []

        async def queue_request(question, headers):
            """Send a request that stays in the worker's queue."""
            waiting.append(asyncio.create_task(client.post(CHAT_URL, json=payload(question), headers=headers)))
            async with asyncio.timeout(1):
                while queue.qsize() < len(waiting):
                    await asyncio.sleep(0)

        async def answer(question, headers):
            """Send a request that the gateway must answer without dispatching it."""
            async with asyncio.timeout(1):
                return await client.post(CHAT_URL, json=payload(question), headers=headers)

        try:
            yield SimpleNamespace(client=client, queue=queue, queue_request=queue_request, answer=answer)
        finally:
            for task in waiting:
                task.cancel()
            await asyncio.gather(*waiting, return_exceptions=True)
            cancel_queued_requests(queue, worker)


@pytest.mark.asyncio
@pytest.mark.parametrize(("headers", "deadline_s"), [(BATCH, 30), (INTERACTIVE, 5), ({}, 5)])
async def test_each_request_class_waits_with_its_own_queue_deadline(gateway, monkeypatch, headers, deadline_s):
    deadlines = []

    async def completed_chat(payload, worker, queue, *, timeout_s, **options):
        deadlines.append(timeout_s)
        return JSONResponse({"worker": worker.id})

    monkeypatch.setattr("gateway.main.serve_queued_chat", completed_chat)

    response = await gateway.client.post(CHAT_URL, json=payload("hello"), headers=headers)

    assert response.status_code == 200
    assert deadlines == [deadline_s]


@pytest.mark.asyncio
async def test_batch_is_refused_when_its_share_of_the_queue_is_taken(gateway):
    await gateway.queue_request("sweep 1", BATCH)
    await gateway.queue_request("sweep 2", BATCH)
    before = queue_full_sheds()

    refused = await gateway.answer("sweep 3", BATCH)

    assert refused.status_code == 503
    assert refused.json()["detail"] == "queue_full"
    assert refused.headers["retry-after"] == "7"
    assert queue_full_sheds() == before + 1
    assert gateway.queue.qsize() == 2


@pytest.mark.asyncio
async def test_interactive_turn_still_gets_a_queue_slot_when_batch_is_at_its_limit(gateway):
    await gateway.queue_request("sweep 1", BATCH)
    await gateway.queue_request("sweep 2", BATCH)

    await gateway.queue_request("student turn", INTERACTIVE)

    assert gateway.queue.qsize() == 3


@pytest.mark.asyncio
async def test_batch_is_refused_when_interactive_turns_already_fill_its_share(gateway):
    await gateway.queue_request("student turn 1", INTERACTIVE)
    await gateway.queue_request("student turn 2", INTERACTIVE)

    refused = await gateway.answer("sweep 1", BATCH)

    # The limit is on queue depth, not on the number of batch requests: with 2
    # requests already waiting, batch must not take one of the last slots.
    assert refused.status_code == 503
    assert refused.json()["detail"] == "queue_full"
