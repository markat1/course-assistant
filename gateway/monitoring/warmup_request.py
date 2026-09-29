from gateway.models.chat.chat_message import ChatMessage
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.settings import Settings

PASSAGE = (
    "Course passage: A KV cache stores attention keys and values for every "
    "previous token so decoding does not recompute them. Its size grows with "
    "layers, KV heads, head dimension, data type and sequence length. "
)
WARMUP_PREFIX = "You are the Course Tutor. \n" + PASSAGE * 40
EXTRA_QUESTIONS = (
    "What is continuous batching?",
    "Why is decode limited by memory bandwidth?",
    "What does a tokenizer do?",
    "What is tensor parallelism?",
)


def build_warmup_request(settings: Settings, question: str | None = None) -> ChatRequest:
    """Build one bounded warmup request on the shared application prefix."""
    return ChatRequest(
        model=settings.model_name,
        messages=[
            ChatMessage(role="system", content=WARMUP_PREFIX),
            ChatMessage(role="user", content=question or settings.warmup_prompt),
        ],
        max_tokens=settings.warmup_max_tokens,
        stream=False,
    )


def build_warmup_requests(settings: Settings) -> list[ChatRequest]:
    """Build warmup requests that share the prefix but ask different questions."""
    questions = [settings.warmup_prompt, *EXTRA_QUESTIONS]
    return [build_warmup_request(settings, question) for question in questions]