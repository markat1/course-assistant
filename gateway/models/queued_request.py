from asyncio import Event, Future
from dataclasses import dataclass, field
from itertools import count

from httpx import Response
from gateway.models.chat.chat_request import ChatRequest

_arrivals = count()

@dataclass
class QueuedRequest:
    """A request awaiting dispatch to its selected worker."""
    payload: ChatRequest
    result: Future[Response]
    expires_at: float
    started_at: float | None = None
    finished: Event = field(default_factory=Event)
    priority: int = 0
    sequence: int = field(default_factory=lambda: next(_arrivals))

    def __lt__(self, other: "QueuedRequest") -> bool:
        return (self.priority, self.sequence) < (other.priority, other.sequence)