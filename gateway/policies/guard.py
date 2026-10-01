from dataclasses import dataclass
from gateway.models.chat.chat_request import ChatRequest
from gateway.policies.token_count import (
    MAX_CHARS_PER_TOKEN,
    TokenCounter,
    count_prompt_tokens,
    estimate_tokens,
)

@dataclass(frozen=True)
class Guard:
    ok: bool
    status: int = 200
    reason: str = "ok"
    prompt_tokens: int = 0

def inspect(
        payload: ChatRequest,
        *,
        context_length: int,
        max_output_tokens: int,
        counter: TokenCounter = estimate_tokens,
) -> Guard:
    """Reject requests the engine cannot serve before they use a queue slot."""
    if payload.max_tokens is not None and payload.max_tokens > max_output_tokens:
        return Guard(ok=False, status=400, reason="bad_max_tokens")

    output_tokens = payload.max_tokens or max_output_tokens
    prompt_tokens = count_prompt_tokens(
        payload, counter, max_chars=context_length * MAX_CHARS_PER_TOKEN
    )
    if prompt_tokens + output_tokens > context_length:
        return Guard(ok=False, status=400, reason="prompt_too_long", prompt_tokens=prompt_tokens)

    return Guard(ok=True, prompt_tokens=prompt_tokens)
