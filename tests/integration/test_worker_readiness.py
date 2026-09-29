import asyncio

import httpx
import pytest

from gateway.monitoring.readiness import prepare_worker
from gateway.policies.admission import admit


@pytest.fixture
def engine_responses():
    return {
        ("GET", "/health"): httpx.Response(200),
        ("GET", "/v1/models"): httpx.Response(
            200, json={"data": [{"id": "test-model"}]}
        ),
        ("POST", "/v1/chat/completions"): httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Hello"}}]
            },
        ),
        ("GET", "/metrics"): httpx.Response(
            200,
            text=(
                "sglang:num_running_reqs 0\n"
                "sglang:num_queue_reqs 0\n"
                "sglang:full_token_usage 0.1\n"
            ),
        ),
    }


@pytest.mark.asyncio
async def test_worker_becomes_ready_only_after_all_startup_checks(
    worker, chat_payload, engine_responses
):
    worker.ready = True
    requests = []

    def engine(request):
        assert worker.ready is False
        key = request.method, request.url.path
        requests.append(key)
        return engine_responses[key]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        await prepare_worker(
            client,
            worker,
            [chat_payload],
            warmup_timeout_s=1.0,
            max_metrics_age_s=10.0,
        )

    assert requests == [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("GET", "/metrics"),
    ]
    assert worker.ready is True
    assert worker.engine_running == 0
    assert worker.engine_waiting == 0
    assert worker.kv_cache_usage_ratio == 0.1
    assert worker.metrics_updated_at is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_endpoint", [
    ("GET", "/health"),
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
    ("GET", "/metrics"),
])
async def test_http_failure_stops_preparation_and_clears_readiness(
    worker, chat_payload, engine_responses, failed_endpoint
):
    worker.ready = True
    requests = []
    engine_responses[failed_endpoint] = httpx.Response(503)

    def engine(request):
        key = request.method, request.url.path
        requests.append(key)
        return engine_responses[key]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await prepare_worker(
                client, worker, [chat_payload],
                warmup_timeout_s=1.0, max_metrics_age_s=10.0,
            )

    expected_order = list(engine_responses)
    assert requests == expected_order[:expected_order.index(failed_endpoint) + 1]
    assert caught.value.response.status_code == 503
    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint,body", [
    pytest.param(("GET", "/v1/models"), b'{"data": []}', id="missing-model"),
    pytest.param(("POST", "/v1/chat/completions"), b'{"choices": []}', id="invalid-warmup"),
    pytest.param(("GET", "/metrics"), b"sglang:num_running_reqs 0\n", id="incomplete-metrics"),
])
async def test_invalid_engine_data_cannot_restore_readiness(
    worker, chat_payload, engine_responses, endpoint, body
):
    worker.ready = True
    requests = []
    engine_responses[endpoint] = httpx.Response(200, content=body)

    def engine(request):
        key = request.method, request.url.path
        requests.append(key)
        return engine_responses[key]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(ValueError):
            await prepare_worker(
                client, worker, [chat_payload],
                warmup_timeout_s=1.0, max_metrics_age_s=10.0,
            )

    assert requests[-1] == endpoint
    assert len(requests) == list(engine_responses).index(endpoint) + 1
    assert worker.ready is False


@pytest.mark.asyncio
async def test_stale_metrics_prevent_readiness_after_successful_warmup(
    worker, chat_payload, engine_responses, monkeypatch
):
    monkeypatch.setattr("gateway.monitoring.collector.monotonic", lambda: 100.0)
    monkeypatch.setattr("gateway.models.workers.worker_state.monotonic", lambda: 111.0)

    def engine(request):
        return engine_responses[request.method, request.url.path]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(ValueError, match="Worker metrics are not fresh"):
            await prepare_worker(
                client, worker, [chat_payload],
                warmup_timeout_s=1.0, max_metrics_age_s=10.0,
            )

    assert worker.metrics_updated_at == 100.0
    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_endpoint", [
    ("GET", "/health"),
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
    ("GET", "/metrics"),
])
async def test_cancellation_stops_preparation_and_keeps_worker_unready(
    worker, chat_payload, engine_responses, blocked_endpoint
):
    worker.ready = True
    started = asyncio.Event()
    stopped = asyncio.Event()
    requests = []

    async def engine(request):
        key = request.method, request.url.path
        requests.append(key)
        if key == blocked_endpoint:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        return engine_responses[key]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        task = asyncio.create_task(prepare_worker(
            client, worker, [chat_payload],
            warmup_timeout_s=10.0, max_metrics_age_s=10.0,
        ))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert stopped.is_set()
    assert requests[-1] == blocked_endpoint
    assert worker.ready is False


@pytest.mark.asyncio
async def test_warmup_timeout_prevents_readiness(
    worker, chat_payload, engine_responses
):
    stopped = asyncio.Event()

    async def engine(request):
        key = request.method, request.url.path
        if key == ("POST", "/v1/chat/completions"):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        return engine_responses[key]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        async with asyncio.timeout(1):
            with pytest.raises(TimeoutError):
                await prepare_worker(
                    client, worker, [chat_payload],
                    warmup_timeout_s=0.02, max_metrics_age_s=10.0,
                )

    assert stopped.is_set()
    assert worker.ready is False


@pytest.mark.asyncio
async def test_failed_worker_can_be_prepared_again_after_engine_recovers(
    worker, chat_payload, engine_responses
):
    requests = []
    warmup_endpoint = ("POST", "/v1/chat/completions")
    successful_response = engine_responses[warmup_endpoint]
    engine_responses[warmup_endpoint] = httpx.Response(503)

    def engine(request):
        key = request.method, request.url.path
        requests.append(key)
        return engine_responses[key]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await prepare_worker(
                client, worker, [chat_payload],
                warmup_timeout_s=1.0, max_metrics_age_s=10.0,
            )
        assert worker.ready is False

        engine_responses[warmup_endpoint] = successful_response
        requests.clear()
        await prepare_worker(
            client, worker, [chat_payload],
            warmup_timeout_s=1.0, max_metrics_age_s=10.0,
        )

    assert requests == list(engine_responses)
    assert worker.ready is True


@pytest.mark.asyncio
async def test_prepared_worker_under_kv_pressure_is_rejected_by_admission(
    worker, chat_payload, engine_responses
):
    engine_responses["GET", "/metrics"] = httpx.Response(200, text=(
        "sglang:num_running_reqs 1\n"
        "sglang:num_queue_reqs 0\n"
        "sglang:full_token_usage 0.95\n"
    ))

    def engine(request):
        return engine_responses[request.method, request.url.path]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        await prepare_worker(
            client, worker, [chat_payload],
            warmup_timeout_s=1.0, max_metrics_age_s=10.0,
        )

    decision = admit([worker], max_metrics_age_s=10.0, kv_usage_limit=0.9)
    assert worker.ready is True
    assert decision.status_code == 503
    assert decision.reason == "kv_pressure"


@pytest.mark.asyncio
async def test_every_warmup_request_is_sent_before_metrics(
    worker, chat_payload, engine_responses
):
    requests = []

    def engine(request):
        key = request.method, request.url.path
        requests.append(key)
        return engine_responses[key]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        await prepare_worker(
            client, worker, [chat_payload, chat_payload, chat_payload],
            warmup_timeout_s=1.0, max_metrics_age_s=10.0,
        )

    assert requests == [
        ("GET", "/health"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/completions"),
        ("GET", "/metrics"),
    ]
    assert worker.ready is True


@pytest.mark.asyncio
async def test_preparation_without_warmup_requests_is_rejected(worker, engine_responses):
    requests = []

    def engine(request):
        requests.append(request)
        return engine_responses[request.method, request.url.path]

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(ValueError):
            await prepare_worker(
                client, worker, [], warmup_timeout_s=1.0, max_metrics_age_s=10.0,
            )

    assert requests == []
    assert worker.ready is False
