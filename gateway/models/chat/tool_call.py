from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from gateway.models.chat.tool_function import ToolFunction

class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    id: str = Field(min_length=1, pattern=r"\S")
    type: Literal["function"]
    function: ToolFunction