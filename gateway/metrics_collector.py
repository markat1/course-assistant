import asyncio
from time import monotonic
from urllib.parse import urljoin

import httpx

from gateway.metrics_parser import parse_worker_metrics
from gateway.models.worker_state import WorkerState

async def collect_worker_metrics(client: httpx.AsyncClient, worker: WorkerState) -> None:
    """Fetch and validate metrics before updating the worker state."""

    started_at = monotonic()
    url = urljoin(str(worker.base_url), "/metrics")

    async with asyncio.timeout(2.0):
        response = await client.get(url)
        response.raise_for_status()
        metrics = parse_worker_metrics(response.text)

    worker.engine_running = metrics.engine_running
    worker.engine_waiting = metrics.engine_waiting
    worker.kv_cache_usage_ratio = metrics.kv_cache_usage_ratio
    worker.metrics_updated_at = started_at