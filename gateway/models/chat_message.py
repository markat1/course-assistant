from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, JsonValue, model_validator, Field

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, JsonValue]] | None = None
    tool_call_id: str | None = Field(default=None, min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def validate_tool_result(self) -> Self:
        if self.role != "tool":
            return self

        if self.tool_call_id is None:
            raise ValueError("Tool messages require tool_call_id")

        if self.content is None:
            raise ValueError("Tool messages require content")

        return self