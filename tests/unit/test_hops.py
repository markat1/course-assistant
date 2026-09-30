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


def eviction_count(cause: str) -> float:
    value = GATEWAY_REGISTRY.get_sample_value("orch_hop_evictions_total", {"cause": cause})
    return value or 0.0


def test_full_ledger_eviction_is_counted_as_capacity(monkeypatch):
    monkeypatch.setattr(hops, "HOP_LEDGER", HopLedger(max_prefixes=1))
    before = eviction_count("capacity")

    hops.record_placement(request("You are the Course Tutor.", "What is prefill?"), "worker-a")
    hops.record_placement(request("You are the Course Router.", "What is prefill?"), "worker-a")

    assert eviction_count("capacity") == before + 1


def test_prefixes_of_a_lost_worker_are_counted_as_worker_lost(monkeypatch):
    monkeypatch.setattr(hops, "HOP_LEDGER", HopLedger(max_prefixes=8))
    hops.record_placement(request("You are the Course Tutor.", "What is prefill?"), "worker-a")
    hops.record_placement(request("You are the Course Router.", "What is prefill?"), "worker-a")
    hops.record_placement(request("Summarise the document.", "Summarise it."), "worker-b")
    before = eviction_count("worker_lost")

    hops.forget_worker("worker-a")

    assert eviction_count("worker_lost") == before + 2


def test_prefix_holders_are_looked_up_by_the_shared_prefix(monkeypatch):
    monkeypatch.setattr(hops, "HOP_LEDGER", HopLedger(max_prefixes=8))
    hops.record_placement(request("You are the Course Tutor.", "What is prefill?"), "worker-b")

    assert hops.prefix_holders(request("You are the Course Tutor.", "What is decode?")) == {"worker-b"}
    assert hops.prefix_holders(request("You are the Course Router.", "What is decode?")) == set()
