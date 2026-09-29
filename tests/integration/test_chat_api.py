import asyncio
import gzip
from contextlib import asynccontextmanager
from time import monotonic
from types import SimpleNamespace

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from gateway.execution.dispatch import dispatch_requests
from gateway.execution.lifecycle import cancel_queued_requests
from gateway.execution.queueing import enqueue_request
from gateway.main import app
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.workers.worker_state import WorkerState


PAYLOAD = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 16,
}
CHAT_URL = "/v1/chat/completions"


@pytest.fixture
def serving(monkeypatch):
    @asynccontextmanager
    async def start(handler, *, consume=True, timeout_s=5):
        worker = WorkerState(
            id="worker-a", base_url="http://worker-a:8000/v1", ready=True,
            engine_running=0, engine_waiting=0, kv_cache_usage_ratio=0.1,
            metrics_updated_at=monotonic(),
        )
        queue = asyncio.Queue(maxsize=1)
        settings = SimpleNamespace(
            metrics_max_age_s=10, kv_usage_limit=0.9, queue_timeout_s=timeout_s,
            capacity_retry_after_s=7, queue_max_size=1,
        )
        monkeypatch.setattr(app.state, "workers", {worker.id: worker}, raising=False)
        monkeypatch.setattr(app.state, "queues", {worker.id: queue}, raising=False)
        monkeypatch.setattr(app.state, "settings", settings, raising=False)
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://gateway"
            ) as client,
        ):
            task = (
                asyncio.create_task(dispatch_requests(upstream, worker, queue))
                if consume else None
            )
            try:
                yield SimpleNamespace(client=client, worker=worker, queue=queue, task=task)
            finally:
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                cancel_queued_requests(queue, worker)

    return start


async def counters(client):
    response = await client.get("/metrics")
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name in ("orch_requests_total", "orch_place_total", "orch_shed_total")
    }


def expected_counters(before, *, reason=None, code=None):
    expected = dict(before)
    keys = [
        ("orch_requests_total", ()),
        ("orch_place_total", (("worker", "worker-a"),)),
    ]
    if reason is not None:
        keys.append(("orch_shed_total", (("code", str(code)), ("reason", reason))))
    for key in keys:
        expected[key] = expected.get(key, 0) + 1
    return expected


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 429, 500, 503, 529])
async def test_worker_response_passes_through_queue_without_retry_or_gateway_shedding(
    serving, status
):
    requests = []
    body = b'{"message":"engine reply"}'

    def engine(request):
        requests.append(request)
        return httpx.Response(
            status, content=body,
            headers={"content-type": "application/json", "retry-after": "3", "x-request-id": "req-1"},
        )

    async with serving(engine) as service:
        before = await counters(service.client)
        response = await service.client.post(CHAT_URL, json=PAYLOAD)
        assert response.status_code == status
        assert response.content == body
        assert response.headers["content-type"] == "application/json"
        assert response.headers["retry-after"] == "3"
        assert response.headers["x-request-id"] == "req-1"
        assert len(requests) == 1
        assert str(requests[0].url) == "http://worker-a:8000/v1/chat/completions"
        assert await counters(service.client) == expected_counters(before)
        await asyncio.wait_for(service.queue.join(), timeout=1)


@pytest.mark.asyncio
async def test_full_queue_returns_503_and_preserves_existing_request(serving):
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200)

    async with serving(engine, consume=False) as service:
        existing = enqueue_request(
            service.queue, service.worker, ChatRequest(**PAYLOAD), timeout_s=60
        )
        before = await counters(service.client)
        response = await service.client.post(CHAT_URL, json=PAYLOAD)
        assert response.status_code == 503
        assert response.json() == {"detail": "queue_full"}
        assert response.headers["retry-after"] == "7"
        assert service.queue.qsize() == 1
        assert not existing.result.done()
        assert requests == []
        assert await counters(service.client) == expected_counters(before, reason="queue_full", code=503)


@pytest.mark.asyncio
async def test_queue_deadline_returns_504_without_worker_call(serving):
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200)

    async with serving(engine, consume=False, timeout_s=0.01) as service:
        before = await counters(service.client)
        response = await asyncio.wait_for(service.client.post(CHAT_URL, json=PAYLOAD), timeout=1)
        assert response.status_code == 504
        assert response.json() == {"detail": "timeout_queue"}
        assert "retry-after" not in response.headers
        assert requests == []
        assert await counters(service.client) == expected_counters(before, reason="timeout_queue", code=504)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,status,detail",
    [(httpx.ReadTimeout, 504, "upstream_timeout")],
)
async def test_upstream_transport_error_is_not_shedding_and_next_request_succeeds(
    serving, error_type, status, detail
):
    requests = []

    def engine(request):
        requests.append(request)
        if len(requests) == 1:
            raise error_type("engine error", request=request)
        return httpx.Response(200, json={"choices": []})

    async with serving(engine) as service:
        before = await counters(service.client)
        response = await service.client.post(CHAT_URL, json=PAYLOAD)
        assert response.status_code == status
        assert response.json() == {"detail": detail}
        assert "retry-after" not in response.headers
        assert await counters(service.client) == expected_counters(before)
        response = await service.client.post(CHAT_URL, json=PAYLOAD)
        assert response.status_code == 200
        assert len(requests) == 2
        assert await counters(service.client) == expected_counters(expected_counters(before))


@pytest.mark.asyncio
async def test_refused_connection_returns_502_then_sheds_until_worker_is_prepared(serving):
    requests = []

    def engine(request):
        requests.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    async with serving(engine) as service:
        response = await service.client.post(CHAT_URL, json=PAYLOAD)
        assert response.status_code == 502
        assert response.json() == {"detail": "upstream_unavailable"}
        assert service.worker.ready is False

        response = await service.client.post(CHAT_URL, json=PAYLOAD)
        assert response.status_code == 503
        assert response.json() == {"detail": "no_eligible_workers"}
        assert response.headers["retry-after"] == "7"
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, httpx.InvalidURL, TimeoutError])
async def test_unexpected_dispatch_error_returns_500_and_next_request_succeeds(
    serving, caplog, error_type
):
    requests = []

    def engine(request):
        requests.append(request)
        if len(requests) == 1:
            raise error_type("internal diagnostic detail")
        return httpx.Response(200, json={"message": "next request succeeded"})

    async with serving(engine) as service:
        before = await counters(service.client)
        response = await asyncio.wait_for(
            service.client.post(CHAT_URL, json=PAYLOAD), timeout=1
        )
        assert response.status_code == 500
        assert response.json() == {"detail": "dispatch_failed"}
        assert "retry-after" not in response.headers
        assert await counters(service.client) == expected_counters(before)
        assert not service.task.done()

        following = await asyncio.wait_for(
            service.client.post(CHAT_URL, json=PAYLOAD), timeout=1
        )
        assert following.status_code == 200
        assert len(requests) == 2
        assert not service.task.done()
        await asyncio.wait_for(service.queue.join(), timeout=1)

    errors = [record for record in caplog.records if record.exc_info]
    assert any(
        record.levelno >= 40 and record.exc_info[0] is error_type
        for record in errors
    )


@pytest.mark.asyncio
async def test_decoded_body_does_not_keep_upstream_compression_or_length_headers(serving):
    body = b'{"choices": [], "padding": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'

    def engine(request):
        return httpx.Response(
            200, content=gzip.compress(body),
            headers={"content-encoding": "gzip", "content-type": "application/json", "connection": "close"},
        )

    async with serving(engine) as service:
        response = await service.client.post(CHAT_URL, json=PAYLOAD)
        assert response.content == body
        assert int(response.headers["content-length"]) == len(body)
        assert "content-encoding" not in response.headers
        assert "connection" not in response.headers
