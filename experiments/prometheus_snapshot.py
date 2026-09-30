"""Read Part 5 answers from a live Prometheus: who waits in our queue vs the engine's."""
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Sample:
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class Series:
    labels: dict[str, str]
    points: list[tuple[float, float]]


@dataclass(frozen=True)
class WorkerQueues:
    gateway_queued: float | None
    gateway_in_flight: float | None
    engine_waiting: float | None
    engine_running: float | None
    engine_retracted: float | None
    kv_usage: float | None


QUEUE_QUERIES = {
    "gateway_queued": "orch_replica_queue_depth",
    "gateway_in_flight": "orch_replica_in_flight",
    "engine_waiting": "sglang:num_queue_reqs",
    "engine_running": "sglang:num_running_reqs",
    "engine_retracted": "sglang:num_retracted_reqs",
    "kv_usage": "sglang:full_token_usage",
}


def _result(client: httpx.Client, path: str, params: dict) -> list[dict]:
    body = client.get(path, params=params).json()
    if body.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {body.get('error', body)}")
    return body["data"]["result"]


def instant(client: httpx.Client, promql: str) -> list[Sample]:
    """Evaluate a PromQL expression now."""
    return [
        Sample(item["metric"], float(item["value"][1]))
        for item in _result(client, "/api/v1/query", {"query": promql})
    ]


def range_series(client: httpx.Client, promql: str, *, start: float, end: float, step: str) -> list[Series]:
    """Evaluate a PromQL expression over a time range, for plots."""
    result = _result(
        client, "/api/v1/query_range",
        {"query": promql, "start": start, "end": end, "step": step},
    )
    return [
        Series(item["metric"], [(float(t), float(v)) for t, v in item["values"]])
        for item in result
    ]


def queue_snapshot(client: httpx.Client) -> dict[str, WorkerQueues]:
    """Gateway queue next to engine queue, per worker; None where a source has no sample."""
    values = {
        field: {sample.labels["worker"]: sample.value for sample in instant(client, promql)}
        for field, promql in QUEUE_QUERIES.items()
    }
    workers = sorted(set().union(*values.values()))
    return {
        worker: WorkerQueues(**{field: values[field].get(worker) for field in QUEUE_QUERIES})
        for worker in workers
    }


def shed_counts(client: httpx.Client, *, window: str = "5m") -> dict[tuple[str, str], float]:
    """Requests shed per (reason, code) during the last window."""
    samples = instant(client, f"sum by (reason, code) (increase(orch_shed_total[{window}]))")
    return {(sample.labels["reason"], sample.labels["code"]): sample.value for sample in samples}
