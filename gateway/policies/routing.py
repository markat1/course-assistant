import random
from collections.abc import Callable, Iterable

from gateway.models.workers.worker_state import WorkerState


from gateway.models.workers.worker_state import WorkerState


def gateway_load(worker: WorkerState, dispatch_max: int | None = None) -> int:
    """Gateway-local queued and dispatched work plus the engine's own backlog."""
    load = worker.gateway_queue_depth + worker.gateway_in_flight + (worker.engine_waiting or 0)
    if dispatch_max is not None and worker.ramp_limit is not None:
        load += max(0, dispatch_max - worker.ramp_limit)
    return load

def select_worker(
        workers: Iterable[WorkerState],
        *,
        max_metrics_age_s: float,
        queue_max_size: int,
        dispatch_max: int | None = None,
        choose: Callable[[list[WorkerState]], WorkerState] = random.choice,
        holders: set[str] | frozenset[str] = frozenset(),
        prefix_load_slack: int = 0,
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
    lowest = min(gateway_load(worker, dispatch_max) for worker in candidates)
    warm = [
        worker for worker in candidates
        if worker.id in holders and gateway_load(worker, dispatch_max) <= lowest + prefix_load_slack
    ]
    if warm:
        candidates = warm
        lowest = min(gateway_load(worker, dispatch_max) for worker in warm)

    return choose([worker for worker in candidates if gateway_load(worker, dispatch_max) == lowest])