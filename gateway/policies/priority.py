REQUEST_PRIORITIES = {"interactive": 0, "batch": 1}

def request_priority(request_class: str) -> int:
    """Lower runs first: a student's turn goes before a background sweep."""
    try:
        return REQUEST_PRIORITIES[request_class]
    except KeyError:
        raise ValueError(f"Unknown request clas: {request_class}") from None
