
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from gateway.models.settings import Settings
from gateway.models.worker_state import WorkerState
from gateway.monitoring.polling import monitor_worker

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for the FastAPI application."""
    settings = Settings()
    app.state.settings = settings

    workers = {
        worker_id: WorkerState(id=worker_id, base_url=url)
        for worker_id, url in settings.worker_urls.items()
    }

    app.state.workers = workers

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