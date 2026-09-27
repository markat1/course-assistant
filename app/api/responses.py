from time import time
from uuid import uuid4

from fastapi import Response
from openai import APIStatusError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

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