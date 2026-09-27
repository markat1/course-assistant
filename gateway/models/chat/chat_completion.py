from pydantic import BaseModel, Field
from gateway.models.chat.chat_completion_choice import ChatCompletionChoice

class ChatCompletion(BaseModel):
    choices: list[ChatCompletionChoice] = Field(min_length=1)