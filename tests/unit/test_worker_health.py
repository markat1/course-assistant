import asyncio

import httpx
import pytest

from gateway.monitoring.health import check_worker_health


@pytest.mark.asyncio
async def test_health_check_rejects_an_unhealthy_worker(worker):
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await check_worker_health(client, worker)

    assert caught.value.response.status_code == 503
    assert [(request.method, str(request.url)) for request in requests] == [
        ("GET", "http://worker-a:8000/health")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [False, True])
async def test_successful_health_check_preserves_readiness_state(worker, ready):
    worker.ready = ready

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        await check_worker_health(client, worker)

    assert worker.ready is ready


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_health_check_propagates_transport_failure_without_retry(worker, error_type):
    requests = []
    error = error_type("worker unavailable")

    def engine(request):
        requests.append(request)
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(error_type) as caught:
            await check_worker_health(client, worker)

    assert caught.value is error
    assert len(requests) == 1
    assert worker.ready is False


@pytest.mark.asyncio
async def test_health_check_bounds_a_stalled_request(worker):
    stopped = asyncio.Event()

    async def stalled_engine(request):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled_engine)) as client:
        async with asyncio.timeout(3):
            with pytest.raises(TimeoutError):
                await check_worker_health(client, worker)

    assert stopped.is_set()
    assert worker.ready is False


@pytest.mark.asyncio
async def test_cancelling_health_check_stops_http_request(worker):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def stalled_engine(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled_engine)) as client:
        task = asyncio.create_task(check_worker_health(client, worker))
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
