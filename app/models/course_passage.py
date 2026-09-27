from pydantic import BaseModel, ConfigDict, Field

class CoursePassage(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, pattern=r"\S")
    title: str = Field(min_length=1, pattern=r"\S")
    content: str = Field(min_length=1, pattern=r"\S")
    source: str = Field(min_length=1, pattern=r"\S")