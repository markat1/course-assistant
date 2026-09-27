from asyncio import Future
from dataclasses import dataclass

from httpx import Response
from gateway.models.chat.chat_request import ChatRequest

@dataclass
class QueuedRequest:
    """A request awaiting dispatch to its selected worker."""
    payload: ChatRequest
    result: Future[Response]
    expires_at: float
    started_at: float | None = None