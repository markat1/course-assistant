import hashlib
import logging

from gateway.models.chat.chat_request import ChatRequest
from gateway.monitoring.metrics import HOP_TOKENS_TOTAL, HOP_TOTAL
from gateway.policies.guard import CHARS_PER_TOKEN
from gateway.policies.hop_ledger import Hop, HopLedger

logger = logging.getLogger(__name__)

HOP_LEDGER = HopLedger()


def prefix_key(payload: ChatRequest) -> tuple[str, int]:
    """Hash of the first message (system prompt / tool schema) and its rough token count."""
    first = payload.messages[0].content
    text = first if isinstance(first, str) else str(first)
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    return digest, len(text) // CHARS_PER_TOKEN + 1


def record_placement(payload: ChatRequest, worker_id: str) -> Hop | None:
    """Record where a prefix was placed and count a hop when it moved worker."""
    prefix, tokens = prefix_key(payload)
    hop = HOP_LEDGER.record(prefix, worker_id, tokens=tokens)
    if hop is not None:
        HOP_TOTAL.labels(src=hop.src, dst=hop.dst, backend=hop.backend).inc()
        HOP_TOKENS_TOTAL.inc(hop.tokens)
        logger.info("KV hop %s -> %s prefix=%s tokens=%d backend=%s", hop.src, hop.dst, hop.prefix, hop.tokens, hop.backend)
    return hop


def forget_worker(worker_id: str) -> None:
    """A worker lost its cache (restart or failure); forget its prefixes."""
    HOP_LEDGER.forget_worker(worker_id)