from time import monotonic

from gateway.models.workers.worker_state import WorkerState
from gateway.policies.routing import select_worker


def ready_worker(worker_id: str, *, queued: int = 0, in_flight: int = 0, waiting: int = 0) -> WorkerState:
    return WorkerState(
        id=worker_id,
        base_url=f"http://{worker_id}:8000/v1",
        ready=True,
        engine_running=0,
        engine_waiting=waiting,
        kv_cache_usage_ratio=0.1,
        metrics_updated_at=monotonic(),
        gateway_queue_depth=queued,
        gateway_in_flight=in_flight,
    )


def first(candidates: list[WorkerState]) -> WorkerState:
    return candidates[0]


def test_requests_in_flight_count_as_load():
    busy = ready_worker("worker-a", in_flight=3)
    idle = ready_worker("worker-b")

    selected = select_worker([busy, idle], max_metrics_age_s=10, queue_max_size=16, choose=first)

    assert selected is idle


def test_worker_with_full_gateway_queue_is_skipped():
    full = ready_worker("worker-a", queued=16)
    busy = ready_worker("worker-b", queued=10, in_flight=8)

    selected = select_worker([full, busy], max_metrics_age_s=10, queue_max_size=16, choose=first)

    assert selected is busy


def test_full_queues_still_select_a_worker_so_enqueue_reports_queue_full():
    a = ready_worker("worker-a", queued=16, in_flight=8)
    b = ready_worker("worker-b", queued=16, in_flight=2)

    selected = select_worker([a, b], max_metrics_age_s=10, queue_max_size=16, choose=first)

    assert selected is b


def test_equal_load_is_decided_by_the_choice_function():
    a = ready_worker("worker-a", in_flight=2)
    b = ready_worker("worker-b", in_flight=2)
    offered = []

    def last(candidates: list[WorkerState]) -> WorkerState:
        offered.extend(candidates)
        return candidates[-1]

    selected = select_worker([a, b], max_metrics_age_s=10, queue_max_size=16, choose=last)

    assert offered == [a, b]
    assert selected is b


def test_engine_waiting_still_counts_as_load():
    backlog = ready_worker("worker-a", waiting=4)
    clear = ready_worker("worker-b", in_flight=2)

    selected = select_worker([backlog, clear], max_metrics_age_s=10, queue_max_size=16, choose=first)

    assert selected is clear


def test_worker_holding_the_prefix_wins_a_tie():
    a = ready_worker("worker-a", in_flight=2)
    b = ready_worker("worker-b", in_flight=2)

    selected = select_worker(
        [a, b], max_metrics_age_s=10, queue_max_size=16,
        holders={"worker-b"}, prefix_load_slack=4, choose=first,
    )

    assert selected is b


def test_worker_holding_the_prefix_wins_within_the_load_slack():
    holder = ready_worker("worker-a", in_flight=4)
    idle = ready_worker("worker-b")

    selected = select_worker(
        [holder, idle], max_metrics_age_s=10, queue_max_size=16,
        holders={"worker-a"}, prefix_load_slack=4, choose=first,
    )

    assert selected is holder


def test_least_loaded_wins_when_the_holder_is_busier_than_the_slack():
    holder = ready_worker("worker-a", in_flight=5)
    idle = ready_worker("worker-b")

    selected = select_worker(
        [holder, idle], max_metrics_age_s=10, queue_max_size=16,
        holders={"worker-a"}, prefix_load_slack=4, choose=first,
    )

    assert selected is idle


def test_holder_with_a_full_queue_is_not_preferred():
    holder = ready_worker("worker-a", queued=16)
    other = ready_worker("worker-b", in_flight=8)

    selected = select_worker(
        [holder, other], max_metrics_age_s=10, queue_max_size=16,
        holders={"worker-a"}, prefix_load_slack=100, choose=first,
    )

    assert selected is other


def test_holder_that_is_not_ready_is_not_preferred():
    holder = ready_worker("worker-a")
    holder.ready = False
    other = ready_worker("worker-b", in_flight=3)

    selected = select_worker(
        [holder, other], max_metrics_age_s=10, queue_max_size=16,
        holders={"worker-a"}, prefix_load_slack=4, choose=first,
    )

    assert selected is other


def test_among_several_holders_the_least_loaded_is_chosen():
    a = ready_worker("worker-a", in_flight=3)
    b = ready_worker("worker-b", in_flight=1)
    c = ready_worker("worker-c")

    selected = select_worker(
        [a, b, c], max_metrics_age_s=10, queue_max_size=16,
        holders={"worker-a", "worker-b"}, prefix_load_slack=4, choose=first,
    )

    assert selected is b
