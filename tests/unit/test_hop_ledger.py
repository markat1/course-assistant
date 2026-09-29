from gateway.policies.hop_ledger import Hop, HopLedger


def test_first_placement_of_a_prefix_is_not_a_hop():
    ledger = HopLedger(max_prefixes=8)

    assert ledger.record("prefix-1", "worker-a", tokens=1600) is None


def test_same_worker_again_is_not_a_hop_because_kv_is_already_there():
    ledger = HopLedger(max_prefixes=8)
    ledger.record("prefix-1", "worker-a", tokens=1600)

    assert ledger.record("prefix-1", "worker-a", tokens=1600) is None


def test_moving_a_prefix_to_another_worker_records_a_recompute_hop():
    ledger = HopLedger(max_prefixes=8)
    ledger.record("prefix-1", "worker-a", tokens=1600)

    hop = ledger.record("prefix-1", "worker-b", tokens=1600)

    assert hop == Hop(src="worker-a", dst="worker-b", prefix="prefix-1", tokens=1600, backend="recompute")


def test_oldest_prefix_is_evicted_when_the_ledger_is_full():
    ledger = HopLedger(max_prefixes=2)
    ledger.record("prefix-1", "worker-a", tokens=10)
    ledger.record("prefix-2", "worker-a", tokens=10)
    ledger.record("prefix-3", "worker-a", tokens=10)

    assert ledger.record("prefix-1", "worker-b", tokens=10) is None
    assert ledger.record("prefix-3", "worker-b", tokens=10) is not None


def test_forgetting_a_worker_prevents_ghost_hops_from_its_lost_cache():
    ledger = HopLedger(max_prefixes=8)
    ledger.record("prefix-1", "worker-a", tokens=10)
    ledger.record("prefix-2", "worker-b", tokens=10)

    assert ledger.forget_worker("worker-a") == 1
    assert ledger.record("prefix-1", "worker-b", tokens=10) is None
    assert ledger.record("prefix-2", "worker-a", tokens=10) is not None
