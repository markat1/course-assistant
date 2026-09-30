import pytest

from gateway.policies.tenant_window import TenantDecision, TenantWindow


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def window(clock):
    return TenantWindow(max_tokens=1000, window_s=10.0, clock=clock)


def test_tenant_under_its_budget_is_admitted(window):
    decision = window.admit("alice", tokens=400)

    assert decision == TenantDecision(ok=True)


def test_tenant_that_would_exceed_its_budget_is_rejected_with_429(window):
    window.admit("alice", tokens=800)

    decision = window.admit("alice", tokens=300)

    assert not decision.ok
    assert (decision.status, decision.reason) == (429, "tenant_tokens")


def test_one_tenant_at_its_limit_does_not_block_another_tenant(window):
    window.admit("alice", tokens=1000)

    assert window.admit("bob", tokens=1000).ok


def test_rejected_request_does_not_use_the_tenants_budget(window):
    window.admit("alice", tokens=800)
    window.admit("alice", tokens=300)

    assert window.admit("alice", tokens=200).ok


def test_tokens_leave_the_window_after_window_seconds(window, clock):
    window.admit("alice", tokens=1000)

    clock.now += 10.0

    assert window.admit("alice", tokens=1000).ok


def test_window_slides_so_only_expired_requests_are_freed(window, clock):
    window.admit("alice", tokens=600)
    clock.now += 6.0
    window.admit("alice", tokens=400)

    clock.now += 5.0

    assert window.admit("alice", tokens=600).ok
    assert not window.admit("alice", tokens=1).ok


def test_retry_after_is_the_whole_seconds_until_enough_tokens_expire(window, clock):
    window.admit("alice", tokens=600)
    clock.now += 2.5
    window.admit("alice", tokens=400)

    decision = window.admit("alice", tokens=500)

    assert decision.retry_after_s == 8
