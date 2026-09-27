import asyncio

import httpx
import pytest
from pydantic import ValidationError

from gateway.monitoring.model import check_worker_model


@pytest.mark.asyncio
async def test_worker_rejects_a_missing_configured_model(worker):
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "other-model"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(ValueError, match="Configured model is not served"):
            await check_worker_model(client, worker, model_name="test-model")

    assert [(request.method, str(request.url)) for request in requests] == [
        ("GET", "http://worker-a:8000/v1/models")
    ]
    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["http://worker-a:8000/v1", "http://worker-a:8000/v1/"])
@pytest.mark.parametrize("ready", [False, True])
async def test_matching_model_preserves_readiness(worker, base_url, ready):
    worker.base_url = base_url
    worker.ready = ready
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200, json={
            "object": "list",
            "data": [
                {"id": "other-model"},
                {"id": "test-model", "object": "model", "owned_by": "vllm"},
            ],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        await check_worker_model(client, worker, model_name="test-model")

    assert [(request.method, str(request.url)) for request in requests] == [
        ("GET", "http://worker-a:8000/v1/models")
    ]
    assert worker.ready is ready


@pytest.mark.asyncio
async def test_empty_model_list_rejects_configured_model(worker):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    ) as client:
        with pytest.raises(ValueError, match="Configured model is not served"):
            await check_worker_model(client, worker, model_name="test-model")

    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    b"not json",
    b"[]",
    b"{}",
    b'{"data": null}',
    b'{"data": {}}',
    b'{"data": [null]}',
    b'{"data": [{}]}',
    b'{"data": [{"id": 42}]}',
    b'{"data": [{"id": ""}]}',
    b'{"data": [{"id": "   "}]}',
])
async def test_invalid_model_response_is_rejected(worker, body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        with pytest.raises(ValidationError):
            await check_worker_model(client, worker, model_name="test-model")

    assert worker.ready is False


@pytest.mark.asyncio
async def test_model_check_preserves_http_failure_without_retry(worker):
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(503, content=b"unavailable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await check_worker_model(client, worker, model_name="test-model")

    assert caught.value.response.status_code == 503
    assert len(requests) == 1
    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_model_check_propagates_transport_failure(worker, error_type):
    requests = []
    error = error_type("worker unavailable")

    def engine(request):
        requests.append(request)
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(error_type) as caught:
            await check_worker_model(client, worker, model_name="test-model")

    assert caught.value is error
    assert len(requests) == 1
    assert worker.ready is False


@pytest.mark.asyncio
async def test_model_check_bounds_a_stalled_request(worker):
    stopped = asyncio.Event()

    async def stalled_engine(request):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled_engine)) as client:
        async with asyncio.timeout(3):
            with pytest.raises(TimeoutError):
                await check_worker_model(client, worker, model_name="test-model")

    assert stopped.is_set()
    assert worker.ready is False


@pytest.mark.asyncio
async def test_cancelling_model_check_stops_http_request(worker):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def stalled_engine(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled_engine)) as client:
        task = asyncio.create_task(check_worker_model(client, worker, model_name="test-model"))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert stopped.is_set()
    assert worker.ready is False
