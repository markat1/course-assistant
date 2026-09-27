import re
from collections.abc import Sequence


from app.models.course_passage import CoursePassage

def search_terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))

def match_score(terms: set[str], passage: CoursePassage) -> int:
    passage_terms = search_terms(f"{passage.title} {passage.content}")
    return len(terms & passage_terms)

def find_passages(
        query: str,
        passages: Sequence[CoursePassage],
        *,
        limit: int = 2,
) -> list[CoursePassage]:
    if limit < 1:
        raise ValueError("Search limit must be positive.")

    terms = search_terms(query)
    scored = [
        (match_score(terms, passage), passage)
        for passage in passages
    ]
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)

    return [passage for score, passage in ranked if score > 0][:limit]