import asyncio
import json
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
from gateway.monitoring.metrics import OVERFLOW_TOTAL, PROMPT_TOKEN_DIFFERENCE
from gateway.policies.overflow import stay_or_leave

from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from gateway.api.streaming import finish_stream, relay_stream

def reject_before_dispatch(
    reason: str,
    status_code: int,
    *,
    retry_after_s: int | None = None,
) -> NoReturn:
    """Record a rejection and return its optional retry guidance."""
    headers = {}
    if retry_after_s is not None:
        headers["Retry-After"] = str(retry_after_s)

    SHED_TOTAL.labels(reason=reason, code=str(status_code)).inc()
    record_overflow_decision(status_code)

    raise HTTPException(
        status_code=status_code,
        detail=reason,
        headers=headers,
    )


def record_overflow_decision(status_code: int) -> None:
    """Count where a failed request would go; no overflow destination is configured."""
    decision = "leave_disabled" if stay_or_leave(status_code) == "leave" else "stay"
    OVERFLOW_TOTAL.labels(decision=decision, code=str(status_code)).inc()



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

def observe_engine_count(response: Response, prompt_tokens: int, request_class: str) -> None:
    """Record how far the gateway's prompt count is from the engine's; streams carry no usage."""
    if response.status_code != 200 or isinstance(response, StreamingResponse):
        return

    try:
        engine_tokens = json.loads(response.body)["usage"]["prompt_tokens"]
    except (ValueError, KeyError, TypeError):
        return

    if isinstance(engine_tokens, int):
        PROMPT_TOKEN_DIFFERENCE.labels(request_class=request_class).observe(prompt_tokens - engine_tokens)


async def serve_queued_chat(
        payload: ChatRequest,
        worker: WorkerState,
        queue: asyncio.Queue[QueuedRequest],
        *,
        timeout_s: float,
        capacity_retry_after_s: int,
        priority: int = 0
) -> Response:
    """Enqueue a selected request and translate its outcome to HTTP."""
    try:
        pending = enqueue_request(
            queue,
            worker,
            payload,
            timeout_s=timeout_s,
            priority=priority
        )

        upstream = await wait_for_response(pending)

    except asyncio.QueueFull:
        reject_before_dispatch("queue_full", 503, retry_after_s=capacity_retry_after_s)
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

    if upstream.status_code != 200:
        record_overflow_decision(upstream.status_code)

    if not payload.stream:
        return to_client_response(upstream)

    if upstream.status_code != 200:
        await upstream.aread()
        await finish_stream(upstream, pending)
        return to_client_response(upstream)

    return StreamingResponse(
        relay_stream(upstream, pending),
        media_type="text/event-stream",
        background=BackgroundTask(finish_stream, upstream, pending),
    )