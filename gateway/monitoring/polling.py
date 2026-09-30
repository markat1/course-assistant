import asyncio
import logging

import httpx

from gateway.monitoring.collector import collect_worker_metrics
from gateway.models.workers.worker_state import WorkerState
from gateway.models.settings import Settings
from gateway.monitoring.readiness import prepare_worker
from gateway.monitoring.warmup_request import build_warmup_requests
from gateway.execution.hops import forget_worker
from gateway.policies.ramp import next_dispatch_limit

logger = logging.getLogger(__name__)


async def refresh_worker(
        client: httpx.AsyncClient,
        worker: WorkerState,
        settings: Settings,
) -> None:
    """Prepare an unready worker or refresh its metrics."""
    if worker.ready:
        await collect_worker_metrics(client, worker)
        if worker.ramp_limit is not None:
            worker.ramp_limit = next_dispatch_limit(
                worker.ramp_limit,
                maximum=settings.dispatch_concurrency_per_worker,
                engine_waiting=worker.engine_waiting or 0,
            )
        return
    payloads = build_warmup_requests(settings)
    await prepare_worker(
        client,
        worker,
        payloads,
        warmup_timeout_s=settings.warmup_timeout_s,
        max_metrics_age_s=settings.metrics_max_age_s
    )
    worker.ramp_limit = 1
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
            forget_worker(worker.id)
            logger.warning(
                "Worker refresh failed for %s: %s",
                worker.id,
                exc,
            )
        await asyncio.sleep(settings.metrics_interval_s)
