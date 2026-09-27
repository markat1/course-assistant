import asyncio
from types import SimpleNamespace

import httpx
import pytest

from gateway.models.settings import Settings
from gateway.monitoring import polling


@pytest.fixture
def settings(worker):
    return Settings(
        _env_file=None,
        model_name="test-model",
        worker_urls={worker.id: str(worker.base_url)},
        metrics_interval_s=7,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "timeout", "invalid_metrics"])
async def test_monitor_clears_readiness_and_repeats_preparation_after_failure(
    worker, settings, monkeypatch, caplog, failure
):
    worker.ready = True
    requests = []
    delays = []

    def engine(request):
        requests.append((request.method, request.url.path))
        if len(requests) == 1:
            assert request.url.path == "/metrics"
            if failure == "timeout":
                raise TimeoutError("Metrics timed out")
            if failure == "invalid_metrics":
                return httpx.Response(200, text="")
            return httpx.Response(503)
        assert not worker.ready
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Ready"}}]
            })
        if request.url.path == "/metrics":
            return httpx.Response(200, text=(
                "vllm:num_requests_running 0\n"
                "vllm:num_requests_waiting 0\n"
                "vllm:kv_cache_usage_perc 0.1\n"
            ))
        pytest.fail(f"Unexpected request: {request.url}")

    async def sleep(delay):
        delays.append(delay)
        if len(delays) == 1:
            assert not worker.ready
        else:
            assert worker.ready
            raise asyncio.CancelledError

    monkeypatch.setattr(polling, "asyncio", SimpleNamespace(sleep=sleep))
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(asyncio.CancelledError):
            await polling.monitor_worker(client, worker, settings)

    assert delays == [7, 7]
    assert requests == [
        ("GET", "/metrics"),
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("GET", "/metrics"),
    ]
    assert any(record.levelno >= 30 and worker.id in record.getMessage()
               for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_monitor_propagates_unexpected_errors_and_cancellation(
    worker, settings, monkeypatch, error_type
):
    worker.ready = True
    error = error_type("Stop monitoring")

    def engine(request):
        raise error

    async def sleep(delay):
        pytest.fail("Must not retry cancellation or an unexpected programming error")

    monkeypatch.setattr(polling, "asyncio", SimpleNamespace(sleep=sleep))
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(error_type) as caught:
            await polling.monitor_worker(client, worker, settings)
    assert caught.value is error
