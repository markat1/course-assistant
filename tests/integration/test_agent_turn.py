import json

import httpx
import pytest
import pytest_asyncio
from agents import MaxTurnsExceeded, OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from gateway.models.chat.chat_request import ChatRequest


@pytest_asyncio.fixture
async def gateway_model():
    requests = []
    responses = iter([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "handoff-1",
                "type": "function",
                "function": {
                    "name": "transfer_to_course_tutor",
                    "arguments": "{}",
                },
            }],
        },
        {"role": "assistant", "content": "KV memory grows as context grows."},
    ])

    def gateway(request):
        assert str(request.url) == "http://gateway:8780/v1/chat/completions"
        body = json.loads(request.content)
        ChatRequest.model_validate(body)
        requests.append(body)
        message = next(responses)
        return httpx.Response(200, json={
            "id": f"completion-{len(requests)}",
            "object": "chat.completion",
            "created": 1,
            "model": "Qwen/Qwen3-8B",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
            }],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as http_client:
        async with AsyncOpenAI(
            base_url="http://gateway:8780/v1",
            api_key="local",
            max_retries=0,
            http_client=http_client,
        ) as client:
            model = OpenAIChatCompletionsModel(
                model="Qwen/Qwen3-8B", openai_client=client
            )
            yield model, requests


@pytest.mark.asyncio
async def test_router_handoff_returns_tutor_answer_through_configured_endpoint(gateway_model):
    from app.conversation import run_turn

    model, requests = gateway_model

    result = await run_turn("Why does KV memory grow?", model=model)

    assert result.final_output == "KV memory grows as context grows."
    assert result.last_agent.name == "Course Tutor"
    assert len(requests) == 2
    assert all(request["model"] == "Qwen/Qwen3-8B" for request in requests)
    assert all(request["max_tokens"] == 256 for request in requests)
    assert any(
        message.get("role") == "tool" and message.get("tool_call_id") == "handoff-1"
        for message in requests[1]["messages"]
    )


@pytest.mark.asyncio
async def test_turn_limit_stops_before_another_model_call(gateway_model):
    from app.conversation import run_turn

    model, requests = gateway_model

    with pytest.raises(MaxTurnsExceeded):
        await run_turn("Explain KV memory.", model=model, max_turns=1)

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_browser_history_reaches_router_and_tutor_without_mutation(gateway_model):
    from app.conversation import run_turn
    from app.models.browser.chat_message import BrowserChatMessage

    model, requests = gateway_model
    history = [
        BrowserChatMessage(role="system", content="Answer in Danish."),
        BrowserChatMessage(role="user", content="What is KV memory?"),
        BrowserChatMessage(
            role="assistant", content="Stored attention keys and values."
        ),
        BrowserChatMessage(role="user", content="Why does it grow?"),
    ]
    original = [message.model_dump() for message in history]

    result = await run_turn(history, model=model)

    assert result.last_agent.name == "Course Tutor"
    assert len(requests) == 2
    for request in requests:
        # Each agent's own instructions precede the supplied conversation.
        assert request["messages"][1:1 + len(history)] == original
    assert [message.model_dump() for message in history] == original
