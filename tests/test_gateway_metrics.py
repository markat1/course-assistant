from types import SimpleNamespace
from time import monotonic

import httpx
import pytest
import pytest_asyncio
from prometheus_client.parser import text_string_to_metric_families

from gateway.main import app
from gateway.models.worker_state import WorkerState


@pytest_asyncio.fixture
async def client(monkeypatch):
    # Supply state without starting background polling or external workers.
    monkeypatch.setattr(app.state, "workers", {}, raising=False)
    monkeypatch.setattr(
        app.state,
        "settings",
        SimpleNamespace(metrics_max_age_s=10.0, kv_usage_limit=0.9),
        raising=False,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as session:
        yield session


async def request_count(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        if family.type == "counter"
        for sample in family.samples
        if sample.name == "orch_requests_total"
    ]
    assert len(samples) == 1
    return samples[0].value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "status"),
    [([{"role": "user", "content": "Hello"}], 503), ([], 422)],
)
async def test_chat_count_includes_admission_and_validation_rejections(
    client, messages, status
):
    before = await request_count(client)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen3-8B", "messages": messages},
    )

    assert response.status_code == status
    if status == 503:
        assert response.json() == {"detail": "no_eligible_workers"}
    assert await request_count(client) == before + 1


@pytest.mark.asyncio
async def test_health_scrapes_and_other_routes_do_not_count_as_chat(client):
    before = await request_count(client)

    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/v1/chat/completions")).status_code == 405
    assert (await client.post("/missing")).status_code == 404
    assert await request_count(client) == before
    assert await request_count(client) == before


async def shed_counts(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    return {
        (sample.labels["reason"], sample.labels["code"]): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == "orch_shed_total"
    }


def ready_worker(kv_usage):
    return WorkerState(
        id="worker-a",
        base_url="http://worker-a:8000/v1",
        ready=True,
        engine_running=0,
        engine_waiting=0,
        kv_cache_usage_ratio=kv_usage,
        metrics_updated_at=monotonic(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kv_usage", "status", "reason"),
    [(None, 503, "no_eligible_workers"), (0.9, 429, "kv_pressure")],
)
async def test_admission_rejection_increments_only_its_shed_series(
    client, kv_usage, status, reason
):
    if kv_usage is not None:
        worker = ready_worker(kv_usage)
        app.state.workers = {worker.id: worker}
    before = await shed_counts(client)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == status
    assert response.json() == {"detail": reason}
    expected = dict(before)
    key = (reason, str(status))
    expected[key] = before.get(key, 0) + 1
    assert await shed_counts(client) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [422, 501])
async def test_validation_and_forwarding_placeholder_do_not_count_as_shedding(client, status):
    worker = ready_worker(0.25)
    app.state.workers = {worker.id: worker}
    before = await shed_counts(client)
    messages = [] if status == 422 else [{"role": "user", "content": "Hello"}]

    response = await client.post(
        "/v1/chat/completions", json={"model": "Qwen/Qwen3-8B", "messages": messages}
    )

    assert response.status_code == status
    assert await shed_counts(client) == before


@pytest.mark.asyncio
async def test_no_worker_after_admission_counts_one_availability_rejection(client, monkeypatch):
    worker = ready_worker(0.25)
    app.state.workers = {worker.id: worker}
    monkeypatch.setattr("gateway.main.select_worker", lambda *args, **kwargs: None)
    before = await shed_counts(client)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "no_eligible_workers"}
    expected = dict(before)
    key = ("no_eligible_workers", "503")
    expected[key] = before.get(key, 0) + 1
    assert await shed_counts(client) == expected
