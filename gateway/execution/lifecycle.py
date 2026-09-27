import asyncio
import httpx

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from gateway.execution.dispatch import dispatch_requests
from gateway.models.queued_request import QueuedRequest
from gateway.models.settings import Settings
from gateway.models.workers.worker_state import WorkerState
from gateway.monitoring.polling import monitor_worker

def start_worker_tasks(
        group: asyncio.TaskGroup,
        client: httpx.AsyncClient,
        worker: WorkerState,
        queue: asyncio.Queue[QueuedRequest],
        settings: Settings,
) -> list[asyncio.Task[None]]:
    """Start monitoring and bounded dispatch consumers for one worker."""

    monitor = group.create_task(
        monitor_worker(client, worker, settings),
        name=f"monitor_{worker.id}"
    )

    consumers = [
        group.create_task(
            dispatch_requests(client, worker, queue),
            name=f"dispatch_{worker.id}_{index}",
        )
        for index in range(settings.dispatch_concurrency_per_worker)

    ]

    return [monitor, *consumers]

def cancel_queued_requests(
        queue: asyncio.Queue[QueuedRequest],
        worker: WorkerState,
) -> None:
    """Cancel remaining queue request after consumers have stopped."""
    while True:
        try:
            pending = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        pending.result.cancel()
        queue.task_done()

    worker.gateway_queue_depth = queue.qsize()

@asynccontextmanager
async def manage_worker_tasks(
    client: httpx.AsyncClient,
    workers: dict[str, WorkerState],
    queues: dict[str, asyncio.Queue[QueuedRequest]],
    settings: Settings,
) -> AsyncGenerator[None, None]:
    """Run worker tasks and clean up active queued requests on exit."""
    try:
        async with asyncio.TaskGroup() as group:
            tasks = []

            try:
                for worker in workers.values():
                    tasks.extend(
                        start_worker_tasks(
                            group,
                            client,
                            worker,
                            queues[worker.id],
                            settings,
                        )
                    )

                yield
            finally:
                for task in tasks:
                    task.cancel()
    finally:
        for worker in workers.values():
            cancel_queued_requests(queues[worker.id], worker)
