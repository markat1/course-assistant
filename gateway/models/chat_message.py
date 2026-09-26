from typing import Literal
from pydantic import BaseModel, ConfigDict, JsonValue

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, JsonValue]] | None = None