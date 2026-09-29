import json

import pytest

from app.api.responses import sse_chunks


async def deltas(*parts):
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_stream_starts_with_the_role_carries_every_delta_and_ends_with_done():
    text = "".join([chunk async for chunk in sse_chunks("Hel", deltas("lo", "!"), "course-assistant")])
    events = [event for event in text.split("\n\n") if event]
    bodies = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]

    assert events[-1] == "data: [DONE]"
    assert bodies[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Hel"}
    assert "".join(body["choices"][0]["delta"].get("content", "") for body in bodies) == "Hello!"
    assert bodies[-1]["choices"][0]["finish_reason"] == "stop"
    assert {body["object"] for body in bodies} == {"chat.completion.chunk"}
    assert {body["model"] for body in bodies} == {"course-assistant"}
    assert len({body["id"] for body in bodies}) == 1
