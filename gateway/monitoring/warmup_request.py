from gateway.models.chat.chat_message import ChatMessage
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.settings import Settings

def build_warmup_request(settings: Settings) -> ChatRequest:
    """Build a bounded request for the initial worker warmup."""
    return ChatRequest(
        model=settings.model_name,
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are a course tutor. Explain inference engineering "
                    "concepts using the supplied course context."
                ),
            ),
            ChatMessage(
                role="user",
                content=settings.warmup_prompt
            ),
        ],
        max_tokens=settings.warmup_max_tokens,
        stream=False,
    )