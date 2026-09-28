from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI


def create_model(model_name: str, client: AsyncOpenAI) -> OpenAIChatCompletionsModel:
    """Connects the agents to a model through an existing API client.

    Args:
        model_name: Model Identifier sent to the gateway.
        client: API client configured to call the gateway.
        The caller owns its lifetime and must close it.
    """
    
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    