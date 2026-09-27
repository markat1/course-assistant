from pathlib import Path

import httpx
import pytest

from gateway.monitoring.collector import collect_worker_metrics


@pytest.fixture
def sglang_sample():
    return (
        Path(__file__).parents[1] / "fixtures" / "sglang-v0.5.20-single-worker.prom"
    ).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_collector_normalizes_sglang_full_attention_metrics(
    worker, sglang_sample, monkeypatch
):
    monkeypatch.setattr("gateway.monitoring.collector.monotonic", lambda: 100.0)

    def engine(request):
        assert request.method == "GET"
        assert str(request.url) == "http://worker-a:8000/metrics"
        return httpx.Response(200, text=sglang_sample)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        await collect_worker_metrics(client, worker)

    assert worker.engine_running == 2
    assert worker.engine_waiting == 3
    assert worker.kv_cache_usage_ratio == 0.25
    assert worker.metrics_updated_at == 100.0
    assert not worker.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing_full_pool", "multiple_series", "invalid_ratio"])
async def test_invalid_scrape_preserves_previous_values_and_timestamp(
    worker, sglang_sample, monkeypatch, problem
):
    worker.engine_running = 7
    worker.engine_waiting = 8
    worker.kv_cache_usage_ratio = 0.4
    worker.metrics_updated_at = 90.0
    before = worker.model_dump()
    monkeypatch.setattr("gateway.monitoring.collector.monotonic", lambda: 100.0)

    if problem == "missing_full_pool":
        sample = "\n".join(
            line for line in sglang_sample.splitlines()
            if not line.startswith("sglang:full_token_usage")
        ) + "\n"
    elif problem == "multiple_series":
        sample = sglang_sample + 'sglang:full_token_usage{model_name="other-model"} 0.1\n'
    else:
        sample = sglang_sample.replace("0.25", "1.2")

    def engine(request):
        return httpx.Response(200, text=sample)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(ValueError):
            await collect_worker_metrics(client, worker)

    assert worker.model_dump() == before
