
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from gateway.models.settings import Settings
from gateway.models.workers.worker_state import WorkerState
from gateway.execution.lifecycle import manage_worker_tasks
from gateway.models.queued_request import QueuedRequest
from gateway.monitoring.logging_config import configure_logging
from gateway.monitoring.metrics import TOKEN_COUNTER_INFO
from gateway.policies.tenant_window import TenantWindow
from gateway.policies.token_count import TokenCounter, build_token_counter, counter_name

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage gateway state, worker tasks and the shared HTTP client."""
    settings = Settings()
    workers = create_workers(settings)
    queues = create_queues(workers, settings.queue_max_size)

    app.state.settings = settings
    app.state.workers = workers
    app.state.queues = queues
    app.state.tenant_window = create_tenant_window(settings)

    configure_logging(settings.log_level)
    app.state.token_counter = create_token_counter(settings)

    async with httpx.AsyncClient(
        timeout=settings.upstream_timeout_s,
        trust_env=False,
    ) as client:
        app.state.client = client

        async with manage_worker_tasks(
            client,
            workers,
            queues,
            settings,
        ):
            yield

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
        worker_id: asyncio.PriorityQueue[QueuedRequest](maxsize=max_size)
        for worker_id in workers
    }

def create_tenant_window(settings: Settings) -> TenantWindow:
    """Create the per-tenant token budget."""
    return TenantWindow(
        max_tokens=settings.tenant_max_tokens,
        window_s=settings.tenant_window_s
    )

def create_token_counter(settings: Settings) -> TokenCounter:
    """Build the configured token counter and export which one is active."""
    counter = build_token_counter(settings)
    TOKEN_COUNTER_INFO.clear()
    TOKEN_COUNTER_INFO.labels(counter=counter_name(counter)).set(1)
    return counter
