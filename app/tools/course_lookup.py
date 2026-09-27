from agents import function_tool

from app.knowledge.corpus import COURSE_PASSAGES
from app.knowledge.search import find_passages
from app.models.course_search_result import CourseSearchResult

@function_tool(failure_error_function=None)
def lookup_course(query:str) -> str:
    """Search course notes using short English keywords
    
    Args:
        query: Keywords describing the course concept to look up.
    """
    result = CourseSearchResult(
        query=query,
        passages=find_passages(query, COURSE_PASSAGES),
    )
    return result.model_dump_json()