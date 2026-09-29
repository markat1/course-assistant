from collections.abc import AsyncIterator

import httpx

from gateway.models.queued_request import QueuedRequest


async def finish_stream(upstream: httpx.Response, pending: QueuedRequest) -> None:
    """Close the engine stream and release the dispatch slot; safe to call twice."""
    await upstream.aclose()
    pending.finished.set()

async def relay_stream(upstream: httpx.Response, pending: QueuedRequest) -> AsyncIterator[bytes]:
    """Pass engine chunks to the client, then close the engine stream."""
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    finally:
        await finish_stream(upstream, pending)