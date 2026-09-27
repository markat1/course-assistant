from typing import Literal
from pydantic import BaseModel, Field

class BrowserChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, pattern=r"\S")