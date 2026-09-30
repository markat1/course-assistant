from prometheus_client import CollectorRegistry, Counter
from prometheus_client import CollectorRegistry, Counter, Gauge

GATEWAY_REGISTRY = CollectorRegistry()

REQUEST_TOTAL = Counter(
    "orch_requests_total",
    "Total POST requests received at the chat completions endpoint.",
    registry=GATEWAY_REGISTRY
)

SHED_TOTAL = Counter(
    "orch_shed_total",
    "Requests rejected before dispatch by admission, availability or queue limits.",

    labelnames=["reason", "code"],
    registry=GATEWAY_REGISTRY
)

PLACE_TOTAL = Counter(
    "orch_place_total",
    "Worker selections made by the gateway before dispatch.",
    labelnames=["worker"],
    registry=GATEWAY_REGISTRY
)

GUARD_REJECTED_TOTAL = Counter(
    "orch_guard_rejected_total",
    "Requests rejected by the guard before admission, by reason.",
    labelnames=["reason"],
    registry=GATEWAY_REGISTRY
)

OVERFLOW_TOTAL = Counter(
    "orch_overflow_total",
    "Stay-or-leave decisions for failed requests; overflow is disabled, so leave is only counted.",
    labelnames=["decision", "code"],
    registry=GATEWAY_REGISTRY
)

HOP_TOTAL = Counter(
    "orch_hop_total",
    "Placements of a known prefix on a different worker; KV is recomputed, not copied.",
    labelnames=["src", "dst", "backend"],
    registry=GATEWAY_REGISTRY
)

HOP_TOKENS_TOTAL = Counter(
    "orch_hop_tokens_total",
    "Estimated prefix tokens recomputed because of hops.",
    registry=GATEWAY_REGISTRY
)

QUEUE_DEPTH = Gauge(
    "orch_replica_queue_depth",
    "Requests waiting in the gateway queue of each worker.",
    labelnames=["worker"],
    registry=GATEWAY_REGISTRY,
)

IN_FLIGHT = Gauge(
    "orch_replica_in_flight",
    "Requests the gateway has dispatched to each worker and not finished.",
    labelnames=["worker"],
    registry=GATEWAY_REGISTRY
)

HOP_EVICTIONS_TOTAL = Counter(
    "orch_hop_evictions_total",
    "Prefixes the hop ledge forgot: capacity (ledger full) or worker_lost (cache gone)",
    labelnames=["cause"],
    registry=GATEWAY_REGISTRY
)