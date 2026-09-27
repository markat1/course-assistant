
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from gateway.models.settings import Settings
from gateway.models.workers.worker_state import WorkerState
from gateway.monitoring.polling import monitor_worker
from gateway.models.queued_request import QueuedRequest

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for the FastAPI application."""

    settings = Settings()
    workers = create_workers(settings)
    queues = create_queues(workers, settings.queue_max_size)

    app.state.settings = settings
    app.state.workers = workers
    app.state.queues = queues

    async with httpx.AsyncClient(timeout=settings.upstream_timeout_s, trust_env=False) as client:
        app.state.client = client

        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(
                    monitor_worker(client, worker, settings.metrics_interval_s),
                    name=f"monitor_worker_{worker.id}"
                )
                for worker in workers.values()
            ]

            try:
                yield
            finally:
                for task in tasks:
                    task.cancel()

def create_workers(settings: Settings) -> dict[str, WorkerState]:
    """Create worker state from configured endpoints."""
    return {
        worker_id: WorkerState(id=worker_id, base_url=url)
        for worker_id, url in settings.worker_urls.items()
    }

def create_queues(
        workers: dict[str, WorkerState],
        max_size: int,
) -> dict[str, asyncio.Queue[QueuedRequest]]:
    """Create one bounded request queue per worker."""
    return {
        worker_id: asyncio.Queue[QueuedRequest](maxsize=max_size)
        for worker_id in workers
    }