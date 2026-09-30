import asyncio
from time import monotonic
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi.responses import JSONResponse
from prometheus_client.parser import text_string_to_metric_families

from gateway.main import app
from gateway.models.workers.worker_state import WorkerState
from gateway.policies.tenant_window import TenantWindow

CHAT_URL = "/v1/chat/completions"
# estimate_prompt_tokens("Hello") = 5 // 4 + 1 = 2, so each request costs 2 + max_tokens.
HELLO = [{"role": "user", "content": "Hello"}]


def payload(max_tokens=None):
    body = {"model": "test-model", "messages": HELLO}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def ready_worker():
    return WorkerState(
        id="worker-a", base_url="http://worker-a:8000/v1", ready=True,
        engine_running=0, engine_waiting=0, kv_cache_usage_ratio=0.1,
        metrics_updated_at=monotonic(),
    )


@pytest_asyncio.fixture
async def gateway(monkeypatch):
    """Gateway with one ready worker and a 1,000-token tenant budget per 60 s."""
    served = []

    async def completed_chat(payload, worker, queue, *, timeout_s, capacity_retry_after_s):
        served.append(payload)
        return JSONResponse({"worker": worker.id})

    monkeypatch.setattr("gateway.main.serve_queued_chat", completed_chat)
    monkeypatch.setattr(app.state, "workers", {"worker-a": ready_worker()}, raising=False)
    monkeypatch.setattr(app.state, "queues", {"worker-a": asyncio.Queue()}, raising=False)
    monkeypatch.setattr(
        app.state, "settings",
        SimpleNamespace(
            metrics_max_age_s=10.0, kv_usage_limit=0.9, queue_timeout_s=5,
            capacity_retry_after_s=7, queue_max_size=16,
            context_length=8192, max_output_tokens=1024,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        app.state, "tenant_window", TenantWindow(max_tokens=1000, window_s=60.0), raising=False,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as client:
        yield SimpleNamespace(client=client, served=served)


async def counters(client):
    response = await client.get("/metrics")
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name in ("orch_shed_total", "orch_overflow_total", "orch_place_total")
    }


def delta(before, after, key):
    return after.get(key, 0) - before.get(key, 0)


@pytest.mark.asyncio
async def test_tenant_over_its_budget_gets_429_with_retry_after(gateway):
    first = await gateway.client.post(CHAT_URL, json=payload(900), headers={"X-Tenant": "alice"})

    second = await gateway.client.post(CHAT_URL, json=payload(200), headers={"X-Tenant": "alice"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "tenant_tokens"
    assert int(second.headers["retry-after"]) > 0
    assert len(gateway.served) == 1


@pytest.mark.asyncio
async def test_one_tenant_at_its_limit_does_not_block_another(gateway):
    await gateway.client.post(CHAT_URL, json=payload(990), headers={"X-Tenant": "alice"})

    response = await gateway.client.post(CHAT_URL, json=payload(990), headers={"X-Tenant": "bob"})

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_requests_without_a_tenant_header_share_the_default_budget(gateway):
    first = await gateway.client.post(CHAT_URL, json=payload(900))

    second = await gateway.client.post(CHAT_URL, json=payload(200))

    assert (first.status_code, second.status_code) == (200, 429)


@pytest.mark.asyncio
async def test_missing_max_tokens_is_charged_as_the_output_limit(gateway, monkeypatch):
    monkeypatch.setattr(
        app.state, "tenant_window", TenantWindow(max_tokens=1030, window_s=60.0), raising=False,
    )

    first = await gateway.client.post(CHAT_URL, json=payload(), headers={"X-Tenant": "alice"})
    second = await gateway.client.post(CHAT_URL, json=payload(16), headers={"X-Tenant": "alice"})

    # 2 + 1024 = 1026 fits in 1030; another 2 + 16 does not.
    assert (first.status_code, second.status_code) == (200, 429)


@pytest.mark.asyncio
async def test_guard_rejection_does_not_use_the_tenants_budget(gateway):
    rejected = await gateway.client.post(CHAT_URL, json=payload(4096), headers={"X-Tenant": "alice"})

    accepted = await gateway.client.post(CHAT_URL, json=payload(990), headers={"X-Tenant": "alice"})

    assert rejected.status_code == 400
    assert accepted.status_code == 200


@pytest.mark.asyncio
async def test_tenant_limit_is_checked_before_worker_admission(gateway, monkeypatch):
    await gateway.client.post(CHAT_URL, json=payload(990), headers={"X-Tenant": "alice"})
    monkeypatch.setattr(app.state, "workers", {}, raising=False)

    response = await gateway.client.post(CHAT_URL, json=payload(100), headers={"X-Tenant": "alice"})

    assert response.status_code == 429
    assert response.json()["detail"] == "tenant_tokens"


@pytest.mark.asyncio
async def test_tenant_rejection_is_counted_as_a_shed_that_stays(gateway):
    await gateway.client.post(CHAT_URL, json=payload(990), headers={"X-Tenant": "alice"})
    before = await counters(gateway.client)

    await gateway.client.post(CHAT_URL, json=payload(100), headers={"X-Tenant": "alice"})

    after = await counters(gateway.client)
    assert delta(before, after, ("orch_shed_total", (("code", "429"), ("reason", "tenant_tokens")))) == 1
    assert delta(before, after, ("orch_overflow_total", (("code", "429"), ("decision", "stay")))) == 1
    assert delta(before, after, ("orch_overflow_total", (("code", "429"), ("decision", "leave_disabled")))) == 0
    assert delta(before, after, ("orch_place_total", (("worker", "worker-a"),))) == 0
