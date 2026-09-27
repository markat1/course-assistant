import json

import httpx
import pytest

from gateway.execution.forwarding import forward_chat
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.workers.worker_state import WorkerState


@pytest.fixture
def payload_body():
    return {
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "Explain admission control."}],
        "max_tokens": 32,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_course",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["http://worker-a:8000/v1", "http://worker-a:8000/v1/"])
async def test_forwarding_preserves_payload_and_joins_worker_url(base_url, payload_body):
    worker = WorkerState(id="worker-a", base_url=base_url)
    payload = ChatRequest.model_validate(payload_body)
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        response = await forward_chat(client, worker, payload)

    assert response.status_code == 200
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://worker-a:8000/v1/chat/completions"
    assert requests[0].headers["content-type"] == "application/json"
    assert json.loads(requests[0].content) == payload_body


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 429, 500, 503, 529])
async def test_forwarding_returns_engine_status_body_and_headers_without_retry(status, payload_body):
    worker = WorkerState(id="worker-a", base_url="http://worker-a:8000/v1")
    payload = ChatRequest.model_validate({**payload_body, "stream": False})
    requests = []
    body = b'{"message":"engine response"}'

    def engine(request):
        requests.append(request)
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            status,
            content=body,
            headers={"content-type": "application/json", "x-request-id": "engine-123"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        response = await forward_chat(client, worker, payload)

    assert response.status_code == status
    assert response.content == body
    assert response.headers["x-request-id"] == "engine-123"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_forwarding_propagates_transport_errors_without_retry(error_type, payload_body):
    worker = WorkerState(id="worker-a", base_url="http://worker-a:8000/v1")
    payload = ChatRequest.model_validate(payload_body)
    requests = []

    def engine(request):
        requests.append(request)
        raise error_type("Engine unavailable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(error_type, match="Engine unavailable"):
            await forward_chat(client, worker, payload)

    assert len(requests) == 1
