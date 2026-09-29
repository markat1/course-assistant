import pytest

from gateway.policies.overflow import stay_or_leave


@pytest.mark.parametrize("status", [429, 500])
def test_client_and_local_failures_stay(status):
    assert stay_or_leave(status) == "stay"


@pytest.mark.parametrize("status", [503, 529])
def test_capacity_failures_may_leave(status):
    assert stay_or_leave(status) == "leave"


@pytest.mark.parametrize("status", [400, 502, 504])
def test_other_failures_stay(status):
    assert stay_or_leave(status) == "stay"
