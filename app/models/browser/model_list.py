from typing import Literal

from openai.types import Model as OpenAIModel
from pydantic import BaseModel, Field


class BrowserModelList(BaseModel):
    object_type: Literal["list"] = Field(default="list", alias="object")
    data: list[OpenAIModel]