import httpx
import pytest
import pytest_asyncio
from prometheus_client.parser import text_string_to_metric_families

from gateway.main import app
from gateway.models.workers.worker_state import WorkerState


def worker(worker_id, *, queued, in_flight):
    return WorkerState(
        id=worker_id, base_url=f"http://{worker_id}:8000/v1",
        gateway_queue_depth=queued, gateway_in_flight=in_flight,
    )


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as session:
        yield session


async def per_worker(client, name):
    response = await client.get("/metrics")
    assert response.status_code == 200
    return {
        sample.labels["worker"]: sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.name == name
    }


@pytest.mark.asyncio
async def test_scrape_reports_gateway_queue_depth_per_worker(client, monkeypatch):
    workers = {
        "worker-a": worker("worker-a", queued=3, in_flight=8),
        "worker-b": worker("worker-b", queued=0, in_flight=1),
    }
    monkeypatch.setattr(app.state, "workers", workers, raising=False)

    depth = await per_worker(client, "orch_replica_queue_depth")
    in_flight = await per_worker(client, "orch_replica_in_flight")

    assert depth == {"worker-a": 3.0, "worker-b": 0.0}
    assert in_flight == {"worker-a": 8.0, "worker-b": 1.0}


@pytest.mark.asyncio
async def test_each_scrape_reads_the_current_depth(client, monkeypatch):
    busy = worker("worker-a", queued=5, in_flight=8)
    monkeypatch.setattr(app.state, "workers", {"worker-a": busy}, raising=False)
    assert (await per_worker(client, "orch_replica_queue_depth"))["worker-a"] == 5.0

    busy.gateway_queue_depth = 0

    assert (await per_worker(client, "orch_replica_queue_depth"))["worker-a"] == 0.0
