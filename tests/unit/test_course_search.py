import pytest


@pytest.fixture
def passages():
    from app.models.course_passage import CoursePassage

    return [
        CoursePassage(
            id="queue",
            title="Queue ownership",
            content="Queue depth influences worker placement.",
            source="demo/queue",
        ),
        CoursePassage(
            id="memory",
            title="Memory budget",
            content="Reserve memory for the KV cache.",
            source="demo/memory",
        ),
        CoursePassage(
            id="capacity",
            title="KV capacity",
            content="KV cache memory grows with resident tokens.",
            source="demo/capacity",
        ),
    ]


def test_search_ranks_matches_and_preserves_their_sources(passages):
    from app.knowledge.search import find_passages

    matches = find_passages("KV cache resident tokens", passages)

    assert [passage.id for passage in matches] == ["capacity", "memory"]
    assert matches[0].source == "demo/capacity"
    assert matches[0].content == "KV cache memory grows with resident tokens."


def test_search_matches_titles_despite_case_and_punctuation(passages):
    from app.knowledge.search import find_passages

    matches = find_passages("OWNERSHIP?!", passages)

    assert [passage.id for passage in matches] == ["queue"]


@pytest.mark.parametrize("query", ["photosynthesis", "", "   ", "?!"])
def test_search_returns_no_evidence_when_nothing_matches(passages, query):
    from app.knowledge.search import find_passages

    assert find_passages(query, passages) == []


def test_limit_keeps_the_best_match_without_reordering_the_corpus(passages):
    from app.knowledge.search import find_passages

    original_order = [passage.id for passage in passages]

    matches = find_passages("KV cache resident tokens", passages, limit=1)

    assert [passage.id for passage in matches] == ["capacity"]
    assert [passage.id for passage in passages] == original_order


def test_zero_limit_is_rejected(passages):
    from app.knowledge.search import find_passages

    with pytest.raises(ValueError, match="positive"):
        find_passages("KV", passages, limit=0)
