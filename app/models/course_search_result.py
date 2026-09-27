from pydantic import BaseModel
from app.models.course_passage import CoursePassage

class CourseSearchResult(BaseModel):
    query: str
    passages: list[CoursePassage]