from prometheus_client import CollectorRegistry, Counter

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