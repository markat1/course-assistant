from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI


def create_model(model_name: str, base_url: str,) -> OpenAIChatCompletionsModel:
    """
    Create an OpenAI chat completions model.

    Args:
        model_name (str): The name of the model to be used.
        base_url (str): The base URL for the OpenAI API.

    Returns:
        OpenAIChatCompletionsModel: An instance of the OpenAI chat completions model.
    """
    client = AsyncOpenAI(base_url=base_url,
                         api_key="local",
                         max_retries=0,
                         timeout=20.0)
    
    return OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    