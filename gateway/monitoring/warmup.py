import asyncio
import httpx

from gateway.execution.forwarding import forward_chat
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.workers.worker_state import WorkerState
from gateway.models.chat.chat_completion import ChatCompletion

async def warmup_worker(
        client: httpx.AsyncClient,
        worker: WorkerState,
        payload: ChatRequest,
        *,
        timeout_s: float
) -> httpx.Response:
    """Send a bounded warmup request without changing readiness."""
    async with asyncio.timeout(timeout_s):
        response = await forward_chat(client, worker, payload)
        response.raise_for_status()

    ChatCompletion.model_validate_json(response.content)
    return response
