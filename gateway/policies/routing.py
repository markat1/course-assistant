import random
from collections.abc import Callable, Iterable

from gateway.models.workers.worker_state import WorkerState


from gateway.models.workers.worker_state import WorkerState


def gateway_load(worker: WorkerState) -> int:
    """Gateway-local queued and dispatched work plus the engine's own backlog."""
    return worker.gateway_queue_depth + worker.gateway_in_flight + (worker.engine_waiting or 0)


def select_worker(
        workers: Iterable[WorkerState],
        *,
        max_metrics_age_s: float,
        queue_max_size: int,
        choose: Callable[[list[WorkerState]], WorkerState] = random.choice,
) -> WorkerState | None:
    """Select a least-loaded eligible worker, preferring workers with queue room."""
    eligible = [
        worker for worker in workers
        if worker.ready
        and worker.metrics_age_s is not None
        and worker.metrics_age_s <= max_metrics_age_s
        and worker.engine_running is not None
        and worker.engine_waiting is not None
        and worker.kv_cache_usage_ratio is not None
    ]
    if not eligible:
        return None

    with_room = [worker for worker in eligible if worker.gateway_queue_depth < queue_max_size]
    candidates = with_room or eligible
    lowest = min(gateway_load(worker) for worker in candidates)
    return choose([worker for worker in candidates if gateway_load(worker) == lowest])