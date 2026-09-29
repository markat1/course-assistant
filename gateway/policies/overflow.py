from typing import Literal

LEAVE_STATUSES = frozenset({503, 529})


def stay_or_leave(status_code: int) -> Literal["stay", "leave"]:
    """429/500 are the client's or a local failure and stay; 503/529 may overflow."""
    return "leave" if status_code in LEAVE_STATUSES else "stay"