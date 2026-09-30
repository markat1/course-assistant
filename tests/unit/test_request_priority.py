import pytest

from gateway.lifespan import create_queues
from gateway.execution.queueing import enqueue_request
from gateway.policies.priority import request_priority


def test_interactive_requests_rank_before_batch():
    assert request_priority("interactive") < request_priority("batch")


def test_unknown_request_class_is_rejected():
    with pytest.raises(ValueError):
        request_priority("urgent")


@pytest.fixture
def queue(worker):
    return create_queues({worker.id: worker}, max_size=4)[worker.id]


@pytest.mark.asyncio
async def test_interactive_request_leaves_the_queue_before_earlier_batch(queue, worker, chat_payload):
    batch = enqueue_request(queue, worker, chat_payload, timeout_s=5, priority=request_priority("batch"))
    interactive = enqueue_request(
        queue, worker, chat_payload, timeout_s=5, priority=request_priority("interactive"),
    )

    first, second = queue.get_nowait(), queue.get_nowait()

    assert (first, second) == (interactive, batch)
    for pending in (batch, interactive):
        queue.task_done()
        pending.result.cancel()


@pytest.mark.asyncio
async def test_requests_of_the_same_class_keep_arrival_order(queue, worker, chat_payload):
    batch = request_priority("batch")
    arrived = [
        enqueue_request(queue, worker, chat_payload, timeout_s=5, priority=batch) for _ in range(3)
    ]

    left = [queue.get_nowait() for _ in range(3)]

    assert left == arrived
    for pending in arrived:
        queue.task_done()
        pending.result.cancel()


@pytest.mark.asyncio
async def test_requests_without_a_priority_are_interactive(queue, worker, chat_payload):
    batch = enqueue_request(queue, worker, chat_payload, timeout_s=5, priority=request_priority("batch"))
    default = enqueue_request(queue, worker, chat_payload, timeout_s=5)

    assert queue.get_nowait() is default
    for pending in (batch, default):
        queue.task_done()
        pending.result.cancel()
