import asyncio
from time import monotonic

from gateway.models.chat_request import ChatRequest
from gateway.models.queued_request import QueuedRequest
from gateway.models.worker_state import WorkerState

def enqueue_request(
        queue: asyncio.Queue[QueuedRequest],
        worker: WorkerState,
        payload: ChatRequest,
        *,
        timeout_s: float) -> QueuedRequest:
    """Enqueue immediately or raise QueueFull when capacity is exhausted"""
    pending = QueuedRequest(
        payload=payload,
        result=asyncio.get_running_loop().create_future(),
        expires_at=monotonic() + timeout_s,
    )

    queue.put_nowait(pending)
    worker.gateway_queue_depth = queue.qsize()

    return pending