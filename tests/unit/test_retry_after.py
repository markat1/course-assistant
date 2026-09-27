import pytest
from fastapi import HTTPException
from prometheus_client import CollectorRegistry, Counter
from pydantic import ValidationError

from gateway.api.chat import reject_before_dispatch
from gateway.models.settings import Settings


@pytest.fixture
def settings_values(monkeypatch):
    monkeypatch.delenv("GATEWAY_CAPACITY_RETRY_AFTER_S", raising=False)
    return {
        "_env_file": None,
        "model_name": "test-model",
        "worker_urls": {"worker-a": "http://worker-a:8000/v1"},
    }


def test_capacity_retry_delay_defaults_to_two_seconds(settings_values):
    assert Settings(**settings_values).capacity_retry_after_s == 2


def test_capacity_retry_delay_is_configurable_from_environment(settings_values, monkeypatch):
    monkeypatch.setenv("GATEWAY_CAPACITY_RETRY_AFTER_S", "7")
    assert Settings(**settings_values).capacity_retry_after_s == 7


@pytest.mark.parametrize("delay", [0, -1, 1.5])
def test_invalid_capacity_retry_delay_is_rejected(settings_values, delay):
    with pytest.raises(ValidationError) as caught:
        Settings(**settings_values, capacity_retry_after_s=delay)
    assert any(error["loc"] == ("capacity_retry_after_s",) for error in caught.value.errors())


@pytest.fixture
def shed_registry(monkeypatch):
    registry = CollectorRegistry()
    counter = Counter("test_shed", "Test shed count", ["reason", "code"], registry=registry)
    monkeypatch.setattr("gateway.api.chat.SHED_TOTAL", counter)
    return registry


@pytest.mark.parametrize("reason,delay", [
    ("kv_pressure", 2),
    ("queue_full", 7),
    ("no_eligible_workers", 3),
])
def test_rejection_includes_retry_delay_and_counts_one_shed(shed_registry, reason, delay):
    with pytest.raises(HTTPException) as caught:
        reject_before_dispatch(reason, 503, retry_after_s=delay)

    assert caught.value.status_code == 503
    assert caught.value.detail == reason
    assert caught.value.headers == {"Retry-After": str(delay)}
    assert shed_registry.get_sample_value(
        "test_shed_total", {"reason": reason, "code": "503"}
    ) == 1


def test_rejection_without_retry_policy_omits_header(shed_registry):
    with pytest.raises(HTTPException) as caught:
        reject_before_dispatch("timeout_queue", 504)

    assert caught.value.status_code == 504
    assert caught.value.detail == "timeout_queue"
    assert "Retry-After" not in (caught.value.headers or {})
    assert shed_registry.get_sample_value(
        "test_shed_total", {"reason": "timeout_queue", "code": "504"}
    ) == 1
