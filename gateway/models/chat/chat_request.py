from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal, Self
from gateway.models.chat.chat_message import ChatMessage

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    model: str = Field(min_length=1, pattern=r"\S")
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_tool_sequence(self) -> Self:
        seen: set[str] = set()
        pending: set[str] = set()

        for message in self.messages:
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError(
                        f"Unexpected tool result: {message.tool_call_id}."
                    )
                pending.remove(message.tool_call_id)
                continue

            if pending:
                raise ValueError(
                    f"Missing tool result before next message: {sorted(pending)}."
                )

            for call in message.tool_calls or []:
                if call.id in seen:
                    raise ValueError(f"Duplicate tool call: {call.id}.")
                seen.add(call.id)
                pending.add(call.id)
            
        if pending:
            raise ValueError(f"Missing tool results: {sorted(pending)}.")
        
        return self

    
