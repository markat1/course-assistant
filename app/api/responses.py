from time import time
from uuid import uuid4

from fastapi import Response
from openai import APIStatusError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from collections.abc import AsyncIterator

import json

def chat_completion_response(
        answer: str,
        model: str,
) -> ChatCompletion:
    return ChatCompletion(
        id=f"chatcmpl-{uuid4().hex}",
        object="chat.completion",
        created=int(time()),
        model=model,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=answer
                ),
            )
        ],
    )


def gateway_error_response(error: APIStatusError) -> Response:
    upstream = error.response
    headers = {
        name: upstream.headers[name]
        for name in ("content-type", "retry-after", "x-request-id")
        if name in upstream.headers
    }

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers
    )

def stream_chunk(completion_id: str, model: str, delta: dict, finish_reason: str | None = None) -> str:
    body = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }

    return f"data: {json.dumps(body)}\n\n"

async def sse_chunks(first: str, rest: AsyncIterator[str], model: str) -> AsyncIterator[str]:
    """OpenAI-compatible stream: role + first text, further deltas, stop, DONE."""
    completion_id = f"chatcmpl-{uuid4().hex}"
    yield stream_chunk(completion_id, model, {"role": "assistant", "content": first})
    async for delta in rest:
        yield stream_chunk(completion_id, model, {"content": delta})
    yield stream_chunk(completion_id, model, {}, "stop")
    yield "data: [DONE]\n\n"
