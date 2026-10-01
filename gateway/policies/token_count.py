import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Callable

from gateway.models.chat.chat_request import ChatRequest
from gateway.models.settings import Settings

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]

CHARS_PER_TOKEN = 4
# A prompt with more characters than this per context token is not tokenized.
MAX_CHARS_PER_TOKEN = 16

# Qwen3 chat template with thinking disabled, as the workers run it (DESIGN.md Part 3).
TEMPLATE_TOKENS_PER_MESSAGE = 5  # <|im_start|> role \n ... <|im_end|> \n
TEMPLATE_TOKENS_REPLY = 7  # <|im_start|> assistant \n <think> \n\n </think> \n\n
TEMPLATE_TOKENS_TOOLS = 76  # the "# Tools" instructions around the schemas
TEMPLATE_TOKENS_PER_TOOL_CALL = 4  # <tool_call> \n ... \n </tool_call>
TEMPLATE_TOKENS_PER_TOOL_RESULT = 4  # <tool_response> \n ... \n </tool_response>


def estimate_tokens(text: str) -> int:
    """Rough token count from text length; needs no tokenizer."""
    return len(text) // CHARS_PER_TOKEN + 1


def counter_name(counter: TokenCounter) -> str:
    """The value of GATEWAY_TOKEN_COUNTER that this counter implements."""
    return "estimate" if counter is estimate_tokens else "tokenizer"


def build_token_counter(settings: Settings) -> TokenCounter:
    """Return the configured counter; the estimate if the tokenizer cannot be loaded."""
    if settings.token_counter == "estimate":
        return estimate_tokens

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(settings.model_name)
    except Exception:
        logger.warning(
            "Tokenizer for %s could not be loaded; counting tokens with the estimate",
            settings.model_name,
            exc_info=True,
        )
        return estimate_tokens

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return cache_by_digest(count)


def cache_by_digest(counter: TokenCounter, max_entries: int = 1024) -> TokenCounter:
    """Count each distinct text once; the shared prefix arrives with every request."""
    counts: OrderedDict[str, int] = OrderedDict()

    def cached(text: str) -> int:
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        if digest in counts:
            counts.move_to_end(digest)
            return counts[digest]

        counts[digest] = tokens = counter(text)
        if len(counts) > max_entries:
            counts.popitem(last=False)
        return tokens

    return cached


def count_prompt_tokens(
        payload: ChatRequest,
        counter: TokenCounter = estimate_tokens,
        *,
        max_chars: int | None = None,
) -> int:
    """Prompt tokens of a request: the estimate, or what the engine's chat template renders."""
    if counter is estimate_tokens:
        return sum(len(text) for text in _string_contents(payload)) // CHARS_PER_TOKEN + 1

    texts = _rendered_texts(payload)
    chars = sum(len(text) for text in texts)
    if max_chars is not None and chars > max_chars:
        return chars // CHARS_PER_TOKEN + 1

    return sum(counter(text) for text in texts) + _template_tokens(payload)


def _string_contents(payload: ChatRequest) -> list[str]:
    return [message.content for message in payload.messages if isinstance(message.content, str)]


def _tools(payload: ChatRequest) -> list:
    tools = (payload.model_extra or {}).get("tools")
    return tools if isinstance(tools, list) else []


def _rendered_texts(payload: ChatRequest) -> list[str]:
    """Every piece of request text that the chat template puts in the prompt."""
    texts = []
    for message in payload.messages:
        if isinstance(message.content, str):
            texts.append(message.content)
        elif message.content is not None:
            texts.extend(
                part["text"] for part in message.content if isinstance(part.get("text"), str)
            )
        for call in message.tool_calls or []:
            texts.append(f'{{"name": "{call.function.name}", "arguments": {call.function.arguments}}}')

    texts.extend("\n" + json.dumps(tool, ensure_ascii=False) for tool in _tools(payload))
    return texts


def _template_tokens(payload: ChatRequest) -> int:
    """Tokens the chat template adds around the request text."""
    tokens = TEMPLATE_TOKENS_PER_MESSAGE * len(payload.messages) + TEMPLATE_TOKENS_REPLY
    for message in payload.messages:
        tokens += TEMPLATE_TOKENS_PER_TOOL_CALL * len(message.tool_calls or [])
        if message.role == "tool":
            tokens += TEMPLATE_TOKENS_PER_TOOL_RESULT

    if _tools(payload):
        tokens += TEMPLATE_TOKENS_TOOLS
        if payload.messages[0].role != "system":
            tokens += TEMPLATE_TOKENS_PER_MESSAGE  # the template adds a system block for the tools

    return tokens
