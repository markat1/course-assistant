from pydantic import BaseModel, ConfigDict, Field, JsonValue
from gateway.models.chat_message import ChatMessage

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    model: str = Field(min_length=1, pattern=r"\S")
    messages: list[ChatMessage] = Field(min_length=1)
