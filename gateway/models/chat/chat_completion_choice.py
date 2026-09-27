from pydantic import BaseModel, field_validator
from gateway.models.chat.chat_message import ChatMessage

class ChatCompletionChoice(BaseModel):
    message: ChatMessage

    @field_validator("message")
    @classmethod
    def validate_output(cls, message: ChatMessage) -> ChatMessage:
        if message.role != "assistant":
            raise ValueError("Expected an assistant message")

        has_text = (
            isinstance(message.content, str)
            and bool(message.content.strip())
        )

        if not has_text and not message.tool_calls:
            raise ValueError("Expected text or tool calls")

        return message