import json

import httpx
import pytest
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from app.conversation import run_turn
from app.knowledge.corpus import COURSE_PASSAGES
from gateway.models.chat.chat_request import ChatRequest


def tool_call(name, arguments, call_id):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


def completion(message, number):
    return {
        "id": f"completion-{number}",
        "object": "chat.completion",
        "created": 1,
        "model": "Qwen/Qwen3-8B",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
        }],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("query, expected_ids", [
    ("capacity", ["kv-capacity"]),
    ("photosynthesis", []),
])
async def test_tutor_receives_actual_lookup_results(query, expected_ids):
    requests = []
    responses = iter([
        tool_call("transfer_to_course_tutor", {}, "handoff-1"),
        tool_call("lookup_course", {"query": query}, "lookup-1"),
        {"role": "assistant", "content": "Test answer after lookup."},
    ])

    def gateway(request):
        assert str(request.url) == "http://gateway:8780/v1/chat/completions"
        body = json.loads(request.content)
        ChatRequest.model_validate(body)
        requests.append(body)
        return httpx.Response(200, json=completion(next(responses), len(requests)))

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
            result = await run_turn(query, model=model)

    assert result.final_output == "Test answer after lookup."
    assert result.last_agent.name == "Course Tutor"
    assert len(requests) == 3
    assert "lookup_course" in {
        tool["function"]["name"] for tool in requests[1]["tools"]
    }
    tool_results = [
        message for message in requests[2]["messages"]
        if message.get("role") == "tool"
        and message.get("tool_call_id") == "lookup-1"
    ]
    assert len(tool_results) == 1
    evidence = json.loads(tool_results[0]["content"])
    assert evidence["query"] == query
    assert [passage["id"] for passage in evidence["passages"]] == expected_ids
    originals = {passage.id: passage for passage in COURSE_PASSAGES}
    for passage in evidence["passages"]:
        assert passage == originals[passage["id"]].model_dump(mode="json")
