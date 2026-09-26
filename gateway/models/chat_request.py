from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    model: str = Field(min_length=1, pattern=r"\S")
    messages: list[dict[str, JsonValue]] = Field(min_length=1)
