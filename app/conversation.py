from collections.abc import AsyncIterator, Sequence
from openai.types.responses import ResponseTextDeltaEvent

from agents import (
    Model,
    ModelSettings,
    RunConfig,
    Runner,
    RunResult,
    TResponseInputItem,
)

from app.agents.router import create_router
from app.agents.tutor import create_tutor
from app.models.browser.chat_message import BrowserChatMessage

def to_agent_input(
        conversation: str | Sequence[BrowserChatMessage],
) -> str | list[TResponseInputItem]:
    if isinstance(conversation, str):
        return conversation

    return [
        {"role": message.role, "content": message.content}
        for message in conversation
    ]

async def run_turn(
        conversation: str | Sequence[BrowserChatMessage],
        *,
        model: Model,
        max_turns: int = 6,
) -> RunResult:
    tutor = create_tutor(model)
    router = create_router(model, tutor)

    return await Runner.run(
        router,
        input=to_agent_input(conversation),
        max_turns=max_turns,
        run_config=RunConfig(
            tracing_disabled=True,
            model_settings=ModelSettings(
                max_tokens=256,
                parallel_tool_calls=False,
            ),
        ),
    )

async def stream_turn(
        conversation: str | Sequence[BrowserChatMessage],
        *,
        model: Model,
        max_turns: int = 6,
) -> AsyncIterator[str]:
    """Run Router -> Tutor -> tools and yield only the answer text as it is generated"""
    tutor = create_tutor(model)
    router = create_router(model, tutor)
    result = Runner.run_streamed(
        router,
        input=to_agent_input(conversation),
        max_turns=max_turns,
        run_config=RunConfig(
            tracing_disabled=True,
            model_settings=ModelSettings(max_tokens=256, parallel_tool_calls=False),
        ),
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            yield event.data.delta