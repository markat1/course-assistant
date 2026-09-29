from dataclasses import dataclass
from gateway.models.chat.chat_request import ChatRequest

CHARS_PER_TOKEN = 4

@dataclass(frozen=True)
class Guard:
    ok: bool
    status: int = 200
    reason: str = "ok"

def estimate_prompt_tokens(payload: ChatRequest) ->int:
    """Rough token count from text length; the gateway has no tokenizer."""
    chars = sum(len(message.content) for message in payload.messages if isinstance(message.content, str))
    return chars // CHARS_PER_TOKEN + 1

def inspect(payload: ChatRequest, *, context_length: int, max_output_tokens: int) -> Guard:
    """Reject requests the engine cannot serve before they use a queue slot."""
    if payload.max_tokens is not None and payload.max_tokens > max_output_tokens:
        return Guard(ok=False, status=400, reason="bad_max_tokens")

    output_tokens = payload.max_tokens or max_output_tokens
    if estimate_prompt_tokens(payload) + output_tokens > context_length:
        return Guard(ok=False, status=400, reason="prompt_too_long")

    return Guard(ok=True)