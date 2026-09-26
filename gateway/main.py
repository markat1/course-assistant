from fastapi import FastAPI, HTTPException, Request
from gateway.lifespan import lifespan
from gateway.models.chat_request import ChatRequest
from gateway.admission import admit
from gateway.routing import select_worker

app = FastAPI(title="Course Assistant Gateway", lifespan=lifespan)

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
        raise HTTPException(
            status_code=decision.status_code,
            detail=decision.reason
        )

    worker = select_worker(
        decision.workers,
        max_metrics_age_s=settings.metrics_max_age_s,
    )

    if worker is None:
        raise HTTPException(status_code=503,detail="no_eligible_workers")

    raise HTTPException(
        status_code=501,
        detail=f"Forwarding to {worker.id} is not implemented yet."
    )