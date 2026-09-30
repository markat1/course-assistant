import asyncio
from time import monotonic
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi.responses import JSONResponse

from gateway.execution import hops
from gateway.main import app
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.workers.worker_state import WorkerState
from gateway.monitoring.metrics import GATEWAY_REGISTRY
from gateway.policies.hop_ledger import HopLedger
from gateway.policies.tenant_window import TenantWindow

CHAT_URL = "/v1/chat/completions"
TUTOR = "You are the Course Tutor. Use the course passages."


def body(system, question):
    return {
        "model": "test-model",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "max_tokens": 16,
    }


def ready_worker(worker_id, *, in_flight=0):
    return WorkerState(
        id=worker_id, base_url=f"http://{worker_id}:8000/v1", ready=True,
        engine_running=0, engine_waiting=0, kv_cache_usage_ratio=0.1,
        metrics_updated_at=monotonic(), gateway_in_flight=in_flight,
    )


def hop_count():
    return sum(
        GATEWAY_REGISTRY.get_sample_value(
            "orch_hop_total", {"src": src, "dst": dst, "backend": "recompute"}
        ) or 0.0
        for src, dst in (("worker-a", "worker-b"), ("worker-b", "worker-a"))
    )


@pytest_asyncio.fixture
async def gateway(monkeypatch):
    """Two ready workers, a fresh hop ledger and a fake queue that reports the worker."""
    workers = {"worker-a": ready_worker("worker-a"), "worker-b": ready_worker("worker-b")}

    async def completed_chat(payload, worker, queue, **options):
        return JSONResponse({"worker": worker.id})

    monkeypatch.setattr("gateway.main.serve_queued_chat", completed_chat)
    monkeypatch.setattr(hops, "HOP_LEDGER", HopLedger(max_prefixes=64))
    monkeypatch.setattr(app.state, "workers", workers, raising=False)
    monkeypatch.setattr(
        app.state, "queues", {worker_id: asyncio.Queue() for worker_id in workers}, raising=False,
    )
    monkeypatch.setattr(
        app.state, "settings",
        SimpleNamespace(
            metrics_max_age_s=10.0, kv_usage_limit=0.9, queue_timeout_s=5,
            capacity_retry_after_s=7, queue_max_size=16,
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
        yield SimpleNamespace(client=client, workers=workers)


@pytest.mark.asyncio
async def test_request_follows_its_shared_prefix_to_the_worker_that_holds_it(gateway):
    hops.record_placement(ChatRequest.model_validate(body(TUTOR, "warm")), "worker-b")
    before = hop_count()

    placed = [
        (await gateway.client.post(CHAT_URL, json=body(TUTOR, f"question {n}"))).json()["worker"]
        for n in range(5)
    ]

    assert placed == ["worker-b"] * 5
    assert hop_count() == before


@pytest.mark.asyncio
async def test_busy_holder_gives_way_to_an_idle_worker_and_that_is_a_hop(gateway):
    hops.record_placement(ChatRequest.model_validate(body(TUTOR, "warm")), "worker-b")
    gateway.workers["worker-b"].gateway_in_flight = 8
    before = hop_count()

    response = await gateway.client.post(CHAT_URL, json=body(TUTOR, "question"))

    assert response.json()["worker"] == "worker-a"
    assert hop_count() == before + 1


@pytest.mark.asyncio
async def test_gateway_does_not_slam_a_worker_that_is_still_ramping(gateway):
    gateway.workers["worker-a"].gateway_in_flight = 3
    gateway.workers["worker-b"].ramp_limit = 1

    response = await gateway.client.post(CHAT_URL, json=body("Summarise the document.", "question"))

    assert response.json()["worker"] == "worker-a"
