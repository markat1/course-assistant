import json
from contextlib import asynccontextmanager

import httpx
import pytest
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from gateway.models.chat.chat_request import ChatRequest


def tool_message(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def model_response(message):
    return httpx.Response(200, json={
        "id": "completion-test",
        "object": "chat.completion",
        "created": 1,
        "model": "Qwen/Qwen3-8B",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
        }],
    })


@pytest.fixture
def browser_api():
    @asynccontextmanager
    async def start(handler, *, max_turns=6):
        from app.main import create_app

        requests = []

        def gateway(request):
            assert str(request.url) == "http://gateway:8780/v1/chat/completions"
            body = json.loads(request.content)
            ChatRequest.model_validate(body)
            requests.append(body)
            return handler(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as upstream:
            async with AsyncOpenAI(
                base_url="http://gateway:8780/v1",
                api_key="local",
                max_retries=0,
                http_client=upstream,
            ) as sdk_client:
                model = OpenAIChatCompletionsModel(
                    model="Qwen/Qwen3-8B", openai_client=sdk_client
                )
                app = create_app(model, max_turns=max_turns)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://app"
                ) as browser:
                    yield browser, requests

    return start


def browser_payload():
    return {
        "model": "course-assistant",
        "messages": [{"role": "user", "content": "Explain KV capacity."}],
        "stream": False,
    }


@pytest.mark.asyncio
async def test_browser_can_discover_the_application_without_model_inference(browser_api):
    def unexpected_call(request):
        pytest.fail("Model discovery must not call the engine")

    async with browser_api(unexpected_call) as (browser, requests):
        response = await browser.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [item["id"] for item in body["data"]] == ["course-assistant"]
    assert body["data"][0]["object"] == "model"
    assert requests == []


@pytest.mark.asyncio
async def test_browser_chat_runs_router_tutor_and_actual_lookup(browser_api):
    messages = iter([
        tool_message("transfer_to_course_tutor", {}, "handoff-1"),
        tool_message("lookup_course", {"query": "capacity"}, "lookup-1"),
        {"role": "assistant", "content": "Test answer with course evidence."},
    ])
    payload = browser_payload()
    payload["messages"] = [
        {"role": "user", "content": "What is KV memory?"},
        {"role": "assistant", "content": "Stored attention keys and values."},
        {"role": "user", "content": "How does it limit capacity?"},
    ]

    async with browser_api(lambda request: model_response(next(messages))) as (browser, requests):
        response = await browser.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "course-assistant"
    assert body["id"]
    assert isinstance(body["created"], int)
    assert body["choices"][0]["message"] == {
        "role": "assistant", "content": "Test answer with course evidence."
    }
    assert body["choices"][0]["finish_reason"] == "stop"
    assert len(requests) == 3
    assert requests[0]["messages"][1:] == payload["messages"]
    assert all(request["model"] == "Qwen/Qwen3-8B" for request in requests)
    result = next(
        message for message in requests[2]["messages"]
        if message.get("tool_call_id") == "lookup-1"
    )
    assert json.loads(result["content"])["passages"][0]["id"] == "kv-capacity"


@pytest.mark.asyncio
@pytest.mark.parametrize("status, reason", [(429, "tenant_tokens"), (503, "no_eligible_workers")])
async def test_gateway_rejection_preserves_body_status_and_retry_after(browser_api, status, reason):
    def reject(request):
        return httpx.Response(
            status, json={"detail": reason}, headers={"Retry-After": "7"}
        )

    async with browser_api(reject) as (browser, requests):
        response = await browser.post("/v1/chat/completions", json=browser_payload())

    assert response.status_code == status
    assert response.json() == {"detail": reason}
    assert response.headers["retry-after"] == "7"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type, status, detail", [
    (httpx.ReadTimeout, 504, "gateway_timeout"),
    (httpx.ConnectError, 502, "gateway_unavailable"),
])
async def test_gateway_transport_failure_becomes_http_error(browser_api, error_type, status, detail):
    def fail(request):
        raise error_type("Test failure", request=request)

    async with browser_api(fail) as (browser, requests):
        response = await browser.post("/v1/chat/completions", json=browser_payload())

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_agent_turn_limit_stops_the_browser_request(browser_api):
    def handoff(request):
        return model_response(tool_message("transfer_to_course_tutor", {}, "handoff-1"))

    async with browser_api(handoff, max_turns=1) as (browser, requests):
        response = await browser.post("/v1/chat/completions", json=browser_payload())

    assert response.status_code == 502
    assert response.json() == {"detail": "agent_turn_limit"}
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"stream": True}, {"model": "unknown-model"}])
async def test_unsupported_browser_request_stops_before_inference(browser_api, changes):
    def unexpected_call(request):
        pytest.fail("Invalid requests must not call the engine")

    async with browser_api(unexpected_call) as (browser, requests):
        response = await browser.post(
            "/v1/chat/completions", json={**browser_payload(), **changes}
        )

    assert response.status_code == 422
    assert requests == []
