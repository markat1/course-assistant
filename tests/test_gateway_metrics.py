from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from prometheus_client.parser import text_string_to_metric_families

from gateway.main import app


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
