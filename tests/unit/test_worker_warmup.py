import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from gateway.monitoring.warmup import warmup_worker


@pytest.mark.asyncio
async def test_warmup_rejects_http_success_without_a_completion(worker, chat_payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": []})
        )
    ) as client:
        with pytest.raises(ValidationError):
            await warmup_worker(client, worker, chat_payload, timeout_s=1.0)

    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    pytest.param(b"not json", id="invalid-json"),
    pytest.param(b"[]", id="wrong-root-type"),
    pytest.param(b"{}", id="missing-choices"),
    pytest.param(b'{"choices": null}', id="null-choices"),
    pytest.param(b'{"choices": {}}', id="wrong-choices-type"),
    pytest.param(b'{"choices": [{}]}', id="missing-message"),
])
async def test_warmup_rejects_malformed_completion(worker, chat_payload, body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        with pytest.raises(ValidationError):
            await warmup_worker(client, worker, chat_payload, timeout_s=1.0)

    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    pytest.param({"role": "user", "content": "Hello"}, id="wrong-role"),
    pytest.param({"content": "Hello"}, id="missing-role"),
    pytest.param({"role": "assistant", "content": ""}, id="empty-text"),
    pytest.param({"role": "assistant", "content": "   "}, id="blank-text"),
    pytest.param({"role": "assistant", "content": None}, id="null-without-tools"),
    pytest.param({"role": "assistant", "content": 42}, id="non-string-text"),
    pytest.param({"role": "assistant", "tool_calls": [{}]}, id="malformed-tool-call"),
])
async def test_warmup_rejects_invalid_assistant_output(worker, chat_payload, message):
    body = {"choices": [{"message": message}]}

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    ) as client:
        with pytest.raises(ValidationError):
            await warmup_worker(client, worker, chat_payload, timeout_s=1.0)

    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, ""])
async def test_warmup_accepts_tool_call_without_text(worker, chat_payload, content):
    body = {
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup_course",
                        "arguments": '{"query":"KV cache"}',
                    },
                }],
            },
        }],
    }

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    ) as client:
        response = await warmup_worker(client, worker, chat_payload, timeout_s=1.0)

    assert response.json() == body
    assert worker.ready is False


@pytest.mark.asyncio
async def test_failed_warmup_propagates_engine_error_without_marking_ready(
    worker, chat_payload
):
    requests = []

    def engine(request):
        requests.append(request)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await warmup_worker(client, worker, chat_payload, timeout_s=1.0)

    assert caught.value.response.status_code == 503
    assert [(request.method, str(request.url)) for request in requests] == [
        ("POST", "http://worker-a:8000/v1/chat/completions")
    ]
    assert worker.ready is False


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [False, True])
async def test_successful_warmup_sends_payload_and_preserves_readiness(
    worker, chat_payload, ready
):
    worker.ready = ready
    requests = []
    body = {
        "id": "chatcmpl-warmup",
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello"},
        }],
    }

    def engine(request):
        requests.append(request)
        return httpx.Response(200, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        response = await warmup_worker(client, worker, chat_payload, timeout_s=1.0)

    assert response.status_code == 200
    assert response.json() == body
    assert len(requests) == 1
    assert json.loads(requests[0].content) == chat_payload.model_dump(
        mode="json", exclude_unset=True
    )
    assert worker.ready is ready


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_warmup_propagates_transport_failure_without_retry(
    worker, chat_payload, error_type
):
    requests = []
    error = error_type("worker unavailable")

    def engine(request):
        requests.append(request)
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(engine)) as client:
        with pytest.raises(error_type) as caught:
            await warmup_worker(client, worker, chat_payload, timeout_s=1.0)

    assert caught.value is error
    assert len(requests) == 1
    assert worker.ready is False


@pytest.mark.asyncio
async def test_warmup_timeout_stops_a_stalled_request(worker, chat_payload):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def stalled_engine(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled_engine)) as client:
        async with asyncio.timeout(1):
            with pytest.raises(TimeoutError):
                await warmup_worker(client, worker, chat_payload, timeout_s=0.02)

    assert started.is_set()
    assert stopped.is_set()
    assert worker.ready is False


@pytest.mark.asyncio
async def test_cancelling_warmup_stops_http_request(worker, chat_payload):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def stalled_engine(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled_engine)) as client:
        task = asyncio.create_task(
            warmup_worker(client, worker, chat_payload, timeout_s=10.0)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert stopped.is_set()
    assert worker.ready is False
