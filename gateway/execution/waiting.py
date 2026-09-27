import asyncio
from time import monotonic

import httpx

from gateway.models.queued_request import QueuedRequest

def expire_queued_request(pending: QueuedRequest) -> None:
    """Expire unfinished work that has not started dispatch."""
    if pending.started_at is not None or pending.result.done():
        return

    pending.result.set_exception(
        TimeoutError("Queue deadline exceeded")
    )

async def wait_for_response(pending: QueuedRequest) -> httpx.Response:
    """Await the response while enforcing the queue deadline."""
    remaining = max(0.0, pending.expires_at - monotonic())
    timer = asyncio.get_running_loop().call_later(
        remaining,
        expire_queued_request,
        pending
    )

    try:
        return await pending.result
    finally:
        timer.cancel()