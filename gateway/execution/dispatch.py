import asyncio
from time import monotonic

import httpx
import logging

from gateway.execution.forwarding import forward_chat
from gateway.models.queued_request import QueuedRequest
from gateway.models.workers.worker_state import WorkerState
from gateway.execution.errors import DispatchError

logger = logging.getLogger(__name__)

async def dispatch_requests(
    client: httpx.AsyncClient,
    worker: WorkerState,
    queue: asyncio.Queue[QueuedRequest],
) -> None:
    """Consume requests from the worker's queue."""
    while True:
        pending = await queue.get()
        worker.gateway_queue_depth = queue.qsize()

        try:
            await dispatch_one(client, worker, pending)
        finally:
            queue.task_done()


async def dispatch_one(
    client: httpx.AsyncClient,
    worker: WorkerState,
    pending: QueuedRequest,
) -> None:
    """Dispatch one request and deliver its result or error."""
    if pending.result.done():
        return

    now = monotonic()

    if now >= pending.expires_at:

        pending.result.set_exception(
            TimeoutError("Queue deadline exceeded")
        )
        return

    pending.started_at = now

    try:
        await forward_until_done(client, worker, pending)
    except httpx.HTTPError as exc:
        if not pending.result.done():
            pending.result.set_exception(exc)
    except asyncio.CancelledError:
        pending.result.cancel()
        raise
    except Exception:
        logger.exception(
            "Unexpected dispatch failure for worker %s",
            worker.id,
        )
        if not pending.result.done():
            pending.result.set_exception(
                DispatchError("Request dispatch failed")
            )


async def forward_until_done(
    client: httpx.AsyncClient,
    worker: WorkerState,
    pending: QueuedRequest,
) -> None:
    """Forward a request and cancel upstream work if no longer needed."""
    upstream = asyncio.create_task(
        forward_chat(client, worker, pending.payload)
    )

    try:
        await asyncio.wait(
            (upstream, pending.result),
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not pending.result.done():
            pending.result.set_result(upstream.result())
    finally:
        if not upstream.done():
            upstream.cancel()

        await asyncio.gather(upstream, return_exceptions=True)
