import asyncio
import logging

import httpx

from gateway.metrics_collector import collect_worker_metrics
from gateway.models.worker_state import WorkerState

logger = logging.getLogger(__name__)


async def monitor_worker(
        client: httpx.AsyncClient,
        worker: WorkerState,
        interval_s: float
) -> None:
    while True:
        try:
            await collect_worker_metrics(client, worker)
        except (httpx.HTTPError, TimeoutError, ValueError) as exc:
            logger.warning(
                "Metrics collection failed for %s: %s",
                worker.id,
                exc,
            )
        await asyncio.sleep(interval_s)
