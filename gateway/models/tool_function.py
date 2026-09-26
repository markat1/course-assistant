from pydantic import BaseModel, ConfigDict, Field

class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    name: str = Field(min_length=1, pattern=r"\S")
    arguments: str