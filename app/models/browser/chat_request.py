from typing import Literal
from pydantic import BaseModel, Field
from app.models.browser.chat_message import BrowserChatMessage

class BrowserChatRequest(BaseModel):
    model: Literal["course-assistant"]
    messages: list[BrowserChatMessage] = Field(min_length=1)
    stream: Literal[False] = False