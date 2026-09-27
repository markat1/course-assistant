import httpx

from gateway.models.chat.chat_request import ChatRequest
from gateway.models.workers.worker_state import WorkerState

async def forward_chat(
        client: httpx.AsyncClient,
        worker: WorkerState,
        payload: ChatRequest
) -> httpx.Response:
    """Forward one non-streaming request to the selected worker."""
    url = f"{str(worker.base_url).rstrip('/')}/chat/completions"

    return await client.post(
        url,
        json=payload.model_dump(mode="json", exclude_unset=True),
    )