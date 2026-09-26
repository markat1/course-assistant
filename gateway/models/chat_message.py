from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, JsonValue, model_validator, Field
from gateway.models.tool_call import ToolCall

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, JsonValue]] | None = None
    tool_call_id: str | None = Field(default=None, min_length=1, pattern=r"\S")
    tool_calls: list[ToolCall] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_tool_result(self) -> Self:
        if self.role != "tool":
            return self

        if self.tool_call_id is None:
            raise ValueError("Tool messages require tool_call_id")

        if self.content is None:
            raise ValueError("Tool messages require content")

        return self

    @model_validator(mode="after")
    def validate_tool_calls_role(self) -> Self:
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool_calls")
        return self