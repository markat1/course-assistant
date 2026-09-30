import math

import httpx
import pytest

from experiments.prometheus_snapshot import (
    Sample,
    Series,
    WorkerQueues,
    instant,
    queue_snapshot,
    range_series,
    shed_counts,
)


def vector(*samples):
    """Prometheus instant-query response: samples are (labels, value) pairs."""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": labels, "value": [1790000000.0, value]} for labels, value in samples],
        },
    }


def prometheus(answers):
    """A fake Prometheus that answers instant queries from a {promql: response} map."""
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        if request.url.path == "/api/v1/query":
            return httpx.Response(200, json=answers.get(request.url.params["query"], vector()))
        if request.url.path == "/api/v1/query_range":
            return httpx.Response(200, json=answers[request.url.params["query"]])
        pytest.fail(f"Unexpected Prometheus request: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://prometheus:9090")
    return client, seen


def test_instant_query_returns_labels_and_float_values():
    client, seen = prometheus({
        "orch_replica_queue_depth": vector(({"worker": "worker-a"}, "3"), ({"worker": "worker-b"}, "0")),
    })

    samples = instant(client, "orch_replica_queue_depth")

    assert samples == [Sample({"worker": "worker-a"}, 3.0), Sample({"worker": "worker-b"}, 0.0)]
    assert seen == [{"query": "orch_replica_queue_depth"}]


def test_instant_query_keeps_nan_as_a_float():
    client, _ = prometheus({"sglang:cache_hit_rate": vector(({"worker": "worker-a"}, "NaN"))})

    [sample] = instant(client, "sglang:cache_hit_rate")

    assert math.isnan(sample.value)


def test_failed_query_raises_with_prometheus_error():
    client, _ = prometheus({
        "bad(": {"status": "error", "errorType": "bad_data", "error": "parse error"},
    })

    with pytest.raises(RuntimeError, match="parse error"):
        instant(client, "bad(")


def test_range_query_returns_time_value_points_per_series():
    client, seen = prometheus({
        "orch_replica_queue_depth": {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {"worker": "worker-a"},
                    "values": [[100.0, "1"], [105.0, "4"]],
                }],
            },
        },
    })

    series = range_series(client, "orch_replica_queue_depth", start=100.0, end=105.0, step="5s")

    assert series == [Series({"worker": "worker-a"}, [(100.0, 1.0), (105.0, 4.0)])]
    assert seen == [{"query": "orch_replica_queue_depth", "start": "100.0", "end": "105.0", "step": "5s"}]


def test_queue_snapshot_answers_who_waits_where_per_worker():
    client, _ = prometheus({
        "orch_replica_queue_depth": vector(({"worker": "worker-a"}, "5"), ({"worker": "worker-b"}, "0")),
        "orch_replica_in_flight": vector(({"worker": "worker-a"}, "8"), ({"worker": "worker-b"}, "2")),
        "sglang:num_queue_reqs": vector(
            ({"worker": "worker-a", "job": "sglang"}, "0"), ({"worker": "worker-b", "job": "sglang"}, "0"),
        ),
        "sglang:num_running_reqs": vector(
            ({"worker": "worker-a", "job": "sglang"}, "8"), ({"worker": "worker-b", "job": "sglang"}, "2"),
        ),
        "sglang:num_retracted_reqs": vector(({"worker": "worker-a"}, "0"), ({"worker": "worker-b"}, "0")),
        "sglang:full_token_usage": vector(({"worker": "worker-a"}, "0.02"), ({"worker": "worker-b"}, "0.01")),
    })

    snapshot = queue_snapshot(client)

    assert snapshot == {
        "worker-a": WorkerQueues(
            gateway_queued=5, gateway_in_flight=8, engine_waiting=0, engine_running=8,
            engine_retracted=0, kv_usage=0.02,
        ),
        "worker-b": WorkerQueues(
            gateway_queued=0, gateway_in_flight=2, engine_waiting=0, engine_running=2,
            engine_retracted=0, kv_usage=0.01,
        ),
    }


def test_queue_snapshot_marks_a_worker_missing_from_one_source_as_none():
    client, _ = prometheus({
        "orch_replica_queue_depth": vector(({"worker": "worker-a"}, "0"), ({"worker": "worker-b"}, "0")),
        "orch_replica_in_flight": vector(({"worker": "worker-a"}, "0"), ({"worker": "worker-b"}, "0")),
        "sglang:num_queue_reqs": vector(({"worker": "worker-a"}, "0")),
        "sglang:num_running_reqs": vector(({"worker": "worker-a"}, "1")),
        "sglang:num_retracted_reqs": vector(({"worker": "worker-a"}, "0")),
        "sglang:full_token_usage": vector(({"worker": "worker-a"}, "0.1")),
    })

    snapshot = queue_snapshot(client)

    assert snapshot["worker-b"].engine_running is None
    assert snapshot["worker-b"].gateway_queued == 0


def test_shed_counts_sum_increases_per_reason_over_the_window():
    client, seen = prometheus({
        "sum by (reason, code) (increase(orch_shed_total[10m]))": vector(
            ({"reason": "tenant_tokens", "code": "429"}, "41.7"),
            ({"reason": "queue_full", "code": "503"}, "3"),
        ),
    })

    counts = shed_counts(client, window="10m")

    assert counts == {("tenant_tokens", "429"): 41.7, ("queue_full", "503"): 3.0}
    assert seen == [{"query": "sum by (reason, code) (increase(orch_shed_total[10m]))"}]
