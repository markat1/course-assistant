import httpx
import pytest

from gateway.models.settings import Settings


@pytest.fixture
def settings(worker):
    return Settings(
        _env_file=None,
        model_name="configured-model",
        worker_urls={worker.id: str(worker.base_url)},
        dispatch_concurrency_per_worker=8,
    )


def engine_with_queue(settings, queued):
    """A healthy engine whose waiting queue is read from queued[0] at each scrape."""
    def engine(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.model_name}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Ready"}}]
            })
        if request.url.path == "/metrics":
            return httpx.Response(200, text=(
                "sglang:num_running_reqs 0\n"
                f"sglang:num_queue_reqs {queued[0]}\n"
                "sglang:full_token_usage 0.1\n"
            ))
        pytest.fail(f"Unexpected engine request: {request.url}")
    return engine


@pytest.mark.asyncio
async def test_worker_that_becomes_ready_starts_at_one_and_ramps_per_poll(worker, settings):
    from gateway.monitoring.polling import refresh_worker

    queued = [0]
    limits = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine_with_queue(settings, queued))) as client:
        for _ in range(5):
            await refresh_worker(client, worker, settings)
            limits.append(worker.ramp_limit)

    assert limits == [1, 2, 4, 8, 8]


@pytest.mark.asyncio
async def test_ramp_backs_off_when_the_engine_queues(worker, settings):
    from gateway.monitoring.polling import refresh_worker

    queued = [0]
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine_with_queue(settings, queued))) as client:
        for _ in range(4):
            await refresh_worker(client, worker, settings)
        assert worker.ramp_limit == 8

        queued[0] = 3
        await refresh_worker(client, worker, settings)

    assert worker.ramp_limit == 4


@pytest.mark.asyncio
async def test_worker_that_returns_after_a_failure_ramps_again(worker, settings):
    from gateway.monitoring.polling import refresh_worker

    queued = [0]
    async with httpx.AsyncClient(transport=httpx.MockTransport(engine_with_queue(settings, queued))) as client:
        for _ in range(4):
            await refresh_worker(client, worker, settings)
        assert worker.ramp_limit == 8

        worker.ready = False
        await refresh_worker(client, worker, settings)

    assert worker.ready
    assert worker.ramp_limit == 1
