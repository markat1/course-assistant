import httpx

from gateway.models.chat.chat_request import ChatRequest
from gateway.models.workers.worker_state import WorkerState
from gateway.monitoring.collector import collect_worker_metrics
from gateway.monitoring.health import check_worker_health
from gateway.monitoring.model import check_worker_model
from gateway.monitoring.warmup import warmup_worker

async def prepare_worker(
        client: httpx.AsyncClient,
        worker: WorkerState,
        payload: ChatRequest,
        *,
        warmup_timeout_s: float,
        max_metrics_age_s: float,
) -> None:
    """Mark a worker ready only after all preparation checks succeed."""
    worker.ready = False

    await check_worker_health(client, worker)
    await check_worker_model(client, worker, model_name=payload.model)
    await warmup_worker(
        client,
        worker,
        payload,
        timeout_s=warmup_timeout_s
    )
    await collect_worker_metrics(client, worker)

    age = worker.metrics_age_s
    if age is None or age > max_metrics_age_s:
        raise ValueError("Worker metrics are not fresh")

    worker.ready = True