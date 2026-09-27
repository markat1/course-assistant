from collections.abc import Iterable
from gateway.models.workers.worker_state import WorkerState

def select_worker(workers: Iterable[WorkerState],*,max_metrics_age_s: float) -> WorkerState | None:
    """Select the least-loaded ready worker with fresh, complete metrics."""
    selected = None
    lowest_load = float("inf")

    for worker in workers:
        age = worker.metrics_age_s

        if not worker.ready or age is None or age > max_metrics_age_s:
            continue

        running = worker.engine_running
        waiting = worker.engine_waiting

        if running is None or waiting is None or worker.kv_cache_usage_ratio is None:
            continue

        load = worker.gateway_queue_depth + waiting + running

        if load < lowest_load:
            selected = worker
            lowest_load = load

    return selected