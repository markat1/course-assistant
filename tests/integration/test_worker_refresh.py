import json

import httpx
import pytest

from gateway.models.settings import Settings


@pytest.mark.asyncio
async def test_refresh_prepares_worker_once_then_updates_metrics(worker):
    from gateway.monitoring.polling import refresh_worker

    settings = Settings(
        _env_file=None,
        model_name="configured-model",
        worker_urls={worker.id: str(worker.base_url)},
        warmup_prompt="Explain prefill and decode.",
        warmup_max_tokens=48,
    )
    requests = []
    warmup_bodies = []
    scrapes = 0

    def engine(request):
        nonlocal scrapes
        requests.append((request.method, request.url.path))
        if request.url.path == "/health":
            assert not worker.ready
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.model_name}]})
        if request.url.path == "/v1/chat/completions":
            assert not worker.ready
            warmup_bodies.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Ready"}}]
            })
        if request.url.path == "/metrics":
            scrapes += 1
            return httpx.Response(200, text=(
                f"vllm:num_requests_running {scrapes}\n"
                "vllm:num_requests_waiting 0\n"
                "vllm:kv_cache_usage_perc 0.1\n"
            ))
        pytest.fail(f"Unexpected engine request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        await refresh_worker(client, worker, settings)
        assert worker.ready
        assert worker.engine_running == 1

        await refresh_worker(client, worker, settings)
        assert worker.ready
        assert worker.engine_running == 2

    assert requests == [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("GET", "/metrics"),
        ("GET", "/metrics"),
    ]
    assert len(warmup_bodies) == 1
    assert warmup_bodies[0]["model"] == settings.model_name
    assert warmup_bodies[0]["messages"][-1]["content"] == settings.warmup_prompt
    assert warmup_bodies[0]["max_tokens"] == 48
    assert warmup_bodies[0]["stream"] is False
