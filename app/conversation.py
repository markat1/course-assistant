from agents import Model, ModelSettings, RunConfig, Runner, RunResult

from app.agents.router import create_router
from app.agents.tutor import create_tutor


async def run_turn(
        question:str,
        *,
        model: Model,
        max_turns: int = 6,
) -> RunResult:
    tutor = create_tutor(model)
    router = create_router(model, tutor)

    return await Runner.run(
        router,
        input=question,
        max_turns=max_turns,
        run_config=RunConfig(
            tracing_disabled=True,
            model_settings=ModelSettings(
                max_tokens=256,
                parallel_tool_calls=False,
            ),
        ),
    )