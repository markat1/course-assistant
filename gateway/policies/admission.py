from collections.abc import Iterable

from gateway.models.admission_decision import AdmissionDecision
from gateway.models.workers.worker_state import WorkerState

def admit(
        workers: Iterable[WorkerState],
        *,
        max_metrics_age_s: float,
        kv_usage_limit: float) -> AdmissionDecision:
    """Check worker availability and measured KV pressure."""
    available: list[WorkerState] = []
    eligible: list[WorkerState] = []

    for worker in workers:
        age = worker.metrics_age_s

        if not worker.ready or age is None or age > max_metrics_age_s:
            continue

        if(
            worker.engine_running is None
            or worker.engine_waiting is None
            or worker.kv_cache_usage_ratio is None
        ):
            continue

        available.append(worker)

        if worker.kv_cache_usage_ratio < kv_usage_limit:
            eligible.append(worker)
    
    if eligible:
        return AdmissionDecision(
            status_code=200,
            reason="accepted",
            workers=eligible
        )

    if available:
        return AdmissionDecision(
            status_code=503,
            reason="kv_pressure",
        )

    return AdmissionDecision(
        status_code=503,
        reason="no_eligible_workers"
    )