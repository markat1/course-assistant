from fastapi import FastAPI, HTTPException, Request, Response
from gateway.lifespan import lifespan
from gateway.models.chat_request import ChatRequest
from gateway.admission import admit
from gateway.routing import select_worker
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from gateway.monitoring.metrics import (
    GATEWAY_REGISTRY,
    REQUEST_TOTAL,
    SHED_TOTAL,
    PLACE_TOTAL
)

app = FastAPI(title="Course Assistant Gateway", lifespan=lifespan)

@app.middleware("http")
async def count_chat_requests(request: Request, call_next):
    """Count incoming chat request before request validation."""
    if(
        request.method == "POST"
        and request.url.path == "/v1/chat/completions"
    ):
        REQUEST_TOTAL.inc()

    return await call_next(request)

@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose gateway metrics in Prometheus format."""
    return Response(
        content=generate_latest(GATEWAY_REGISTRY),
        headers={"Content-type": CONTENT_TYPE_LATEST}
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the gateway is running.

    Returns:
        dict: A dictionary indicating the health status of the gateway.
    """
    return {"status": "ok", "service": "gateway"}

@app.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatRequest, request: Request):
    """Validate admission and select a worker for a chat completion"""
    settings = request.app.state.settings
    workers = request.app.state.workers

    decision = admit(
        workers.values(),
        max_metrics_age_s=settings.metrics_max_age_s,
        kv_usage_limit=settings.kv_usage_limit
    )

    if decision.status_code != 200:
        SHED_TOTAL.labels(
            reason=decision.reason,
            code=str(decision.status_code)
        ).inc()

        raise HTTPException(
            status_code=decision.status_code,
            detail=decision.reason
        )

    worker = select_worker(
        decision.workers,
        max_metrics_age_s=settings.metrics_max_age_s,
    )

    if worker is None:
        SHED_TOTAL.labels(
            reason="no_eligible_workers",
            code="503"
        ).inc()

        raise HTTPException(status_code=503,detail="no_eligible_workers")

    PLACE_TOTAL.labels(worker=worker.id).inc()

    raise HTTPException(
        status_code=501,
        detail=f"Forwarding to {worker.id} is not implemented yet."
    )