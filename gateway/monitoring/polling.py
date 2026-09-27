import asyncio
import logging

import httpx

from gateway.monitoring.collector import collect_worker_metrics
from gateway.models.workers.worker_state import WorkerState
from gateway.models.settings import Settings
from gateway.monitoring.readiness import prepare_worker
from gateway.monitoring.warmup_request import build_warmup_request

logger = logging.getLogger(__name__)


async def refresh_worker(
        client: httpx.AsyncClient,
        worker: WorkerState,
        settings: Settings,
) -> None:
    """Prepare an unready worker or refresh its metrics."""
    if worker.ready:
        await collect_worker_metrics(client, worker)
        return
    payload = build_warmup_request(settings)
    await prepare_worker(
        client,
        worker,
        payload,
        warmup_timeout_s=settings.warmup_timeout_s,
        max_metrics_age_s=settings.metrics_max_age_s
    )

async def monitor_worker(
        client: httpx.AsyncClient,
        worker: WorkerState,
        settings: Settings,
) -> None:
    while True:
        try:
            await refresh_worker(client, worker, settings)
        except (httpx.HTTPError, TimeoutError, ValueError) as exc:
            worker.ready = False
            logger.warning(
                "Worker refresh failed for %s: %s",
                worker.id,
                exc,
            )
        await asyncio.sleep(settings.metrics_interval_s)
