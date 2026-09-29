import httpx

from gateway.models.chat.chat_request import ChatRequest
from gateway.models.workers.worker_state import WorkerState

async def forward_chat(
        client: httpx.AsyncClient,
        worker: WorkerState,
        payload: ChatRequest
) -> httpx.Response:
    """Forward one request; a streaming request returns before its body is read."""
    url = f"{str(worker.base_url).rstrip('/')}/chat/completions"
    body = payload.model_dump(mode="json", exclude_unset=True)

    if payload.stream:
        request = client.build_request("POST", url, json=body)
        return await client.send(request, stream=True)

    return await client.post(url, json=body)