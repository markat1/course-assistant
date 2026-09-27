import asyncio
from urllib.parse import urljoin

import httpx

from gateway.models.workers.worker_state import WorkerState

async def check_worker_health(
        client: httpx.AsyncClient,
        worker: WorkerState
) -> None:
    """Check engine health without changing worker readiness."""

    url = urljoin(str(worker.base_url), "/health")

    async with asyncio.timeout(2.0):
        response = await client.get(url)
        response.raise_for_status()
