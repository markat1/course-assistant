from gateway.execution import hops
from gateway.models.chat.chat_request import ChatRequest
from gateway.monitoring.metrics import GATEWAY_REGISTRY
from gateway.policies.hop_ledger import HopLedger


def request(system: str, question: str) -> ChatRequest:
    return ChatRequest.model_validate({
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
    })


def hop_count(src: str, dst: str) -> float:
    value = GATEWAY_REGISTRY.get_sample_value(
        "orch_hop_total", {"src": src, "dst": dst, "backend": "recompute"}
    )
    return value or 0.0


def test_prefix_key_depends_on_the_shared_prefix_not_the_question():
    same_a = hops.prefix_key(request("You are the Course Tutor.", "What is prefill?"))
    same_b = hops.prefix_key(request("You are the Course Tutor.", "What is decode?"))
    other = hops.prefix_key(request("You are the Course Router.", "What is prefill?"))

    assert same_a == same_b
    assert same_a != other


def test_placement_on_another_worker_is_counted_as_a_hop(monkeypatch):
    monkeypatch.setattr(hops, "HOP_LEDGER", HopLedger(max_prefixes=8))
    payload = request("You are the Course Tutor." * 40, "What is prefill?")
    before = hop_count("worker-a", "worker-b")

    assert hops.record_placement(payload, "worker-a") is None
    hop = hops.record_placement(payload, "worker-b")

    assert hop is not None and (hop.src, hop.dst, hop.backend) == ("worker-a", "worker-b", "recompute")
    assert hop.tokens > 200
    assert hop_count("worker-a", "worker-b") == before + 1


def test_forgotten_worker_does_not_produce_a_hop(monkeypatch):
    monkeypatch.setattr(hops, "HOP_LEDGER", HopLedger(max_prefixes=8))
    payload = request("You are the Course Tutor.", "What is prefill?")
    hops.record_placement(payload, "worker-a")

    hops.forget_worker("worker-a")

    assert hops.record_placement(payload, "worker-b") is None
