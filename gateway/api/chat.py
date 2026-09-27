import asyncio
from typing import NoReturn

import httpx
from fastapi import HTTPException, Response

from gateway.execution.queueing import enqueue_request
from gateway.execution.waiting import wait_for_response
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.queued_request import QueuedRequest
from gateway.models.workers.worker_state import WorkerState
from gateway.monitoring.metrics import SHED_TOTAL
from gateway.execution.errors import DispatchError

def reject_before_dispatch(reason: str, status_code: int) -> NoReturn:
    """Record a gateway rejection and return its HTTP error."""
    SHED_TOTAL.labels(reason=reason, code=str(status_code)).inc()
    raise HTTPException(status_code=status_code, detail=reason)

def to_client_response(upstream: httpx.Response) -> Response:
    """Preserve the upstream body, status and selected response headers."""
    headers = {
        name: upstream.headers[name]
        for name in ("content-type", "retry-after", "x-request-id")
        if name in upstream.headers
    }

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
    )

async def serve_queued_chat(
        payload: ChatRequest,
        worker: WorkerState,
        queue: asyncio.Queue[QueuedRequest],
        *,
        timeout_s: float,
) -> Response:
    """Enqueue a selected request and translate its outcome to HTTP."""
    try:
        pending = enqueue_request(
            queue,
            worker,
            payload,
            timeout_s=timeout_s
        )

        upstream = await wait_for_response(pending)

    except asyncio.QueueFull:
        reject_before_dispatch("queue_full", 503)
    except TimeoutError:
        reject_before_dispatch("timeout_queue", 504)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="upstream_timeout",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="upstream_unavailable"
        ) from exc
    except DispatchError as exc:
        raise HTTPException(
            status_code=500,
            detail="dispatch_failed",
        ) from exc

    return to_client_response(upstream)