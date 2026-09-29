from agents import MaxTurnsExceeded
from fastapi import APIRouter, HTTPException, Request, Response
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai.types import Model as OpenAIModel
from openai.types.chat import ChatCompletion

from fastapi.responses import StreamingResponse
from app.api.responses import (
    chat_completion_response,
    gateway_error_response,
      sse_chunks,
)

from app.conversation import run_turn
from app.models.browser.chat_request import BrowserChatRequest
from app.models.browser.model_list import BrowserModelList


from app.conversation import run_turn, stream_turn

router = APIRouter(prefix="/v1")

@router.get(
    "/models",
    response_model=BrowserModelList,
    response_model_by_alias=True,
    response_model_exclude_none=True,
)
async def list_models() -> BrowserModelList:
    return BrowserModelList(
        data=[
            OpenAIModel(
                id="course-assistant",
                object="model",
                created=0,
                owned_by="course-assistant"
            )
        ]
    )

@router.post(
    "/chat/completions",
    response_model=ChatCompletion,
    response_model_exclude_none=True,
)
async def chat_completions(
    payload: BrowserChatRequest,
    request: Request
) -> ChatCompletion | Response:
    if payload.stream:
        return await stream_completion(payload, request)

    try:
        result = await run_turn(
            payload.messages,
            model=request.app.state.model,
            max_turns=request.app.state.max_turns,
        )
    except APITimeoutError as error:
        raise HTTPException(504, detail="gateway_timeout") from error
    except APIConnectionError as error:
        raise HTTPException(502, detail="gateway_unavailable") from error
    except APIStatusError as error:
        return gateway_error_response(error)
    except MaxTurnsExceeded as error:
        raise HTTPException(502, detail="agent_turn_limit") from error

    answer = result.final_output_as(str, raise_if_incorrect_type=True)
    return chat_completion_response(answer, payload.model)

async def stream_completion(payload: BrowserChatRequest, request: Request) -> Response:
    """Wait for the first text so gateway errors become HTTP errors, then stream the rest."""
    deltas = stream_turn(
        payload.messages,
        model=request.app.state.model,
        max_turns=request.app.state.max_turns,
    )
    try:
        first = await anext(deltas, "")
    except APITimeoutError as error:
        raise HTTPException(504, detail="gateway_timeout") from error
    except APIConnectionError as error:
        raise HTTPException(502, detail="gateway_unavailable") from error
    except APIStatusError as error:
        return gateway_error_response(error)
    except MaxTurnsExceeded as error:
        raise HTTPException(502, detail="agent_turn_limit") from error

    return StreamingResponse(sse_chunks(first, deltas, payload.model), media_type="text/event-stream")