from pydantic import BaseModel, Field

class ServedModel(BaseModel):
    id: str = Field(strict=True, min_length=1, pattern=r"\S")
