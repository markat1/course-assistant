from prometheus_client import CollectorRegistry, Counter

GATEWAY_REGISTRY = CollectorRegistry()

REQUEST_TOTAL = Counter(
    "orch_requests_total",
    "Total POST requests received at the chat completions endpoint.",
    registry=GATEWAY_REGISTRY
)

SHED_TOTAL = Counter(
    "orch_shed_total",
    "Request rejected by gateway admission or worker availability.",

    labelnames=["reason", "code"],
    registry=GATEWAY_REGISTRY
)