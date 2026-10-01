import pytest
from pydantic import ValidationError

from gateway.lifespan import create_tenant_window
from gateway.models.settings import Settings


@pytest.fixture
def settings_values():
    return {
        "_env_file": None,
        "model_name": "configured-model",
        "worker_urls": {"worker-a": "http://worker-a:8000/v1"},
    }


def test_tenant_window_uses_the_configured_budget(settings_values):
    settings = Settings(**settings_values, tenant_max_tokens=100, tenant_window_s=60.0)

    window = create_tenant_window(settings)

    assert window.admit("alice", tokens=100).ok
    assert not window.admit("alice", tokens=1).ok


def test_tenant_budget_is_read_from_the_gateway_environment(settings_values, monkeypatch):
    monkeypatch.setenv("GATEWAY_TENANT_MAX_TOKENS", "5000")
    monkeypatch.setenv("GATEWAY_TENANT_WINDOW_S", "30")

    settings = Settings(**settings_values)

    assert settings.tenant_max_tokens == 5000
    assert settings.tenant_window_s == 30.0


@pytest.mark.parametrize("name,value", [
    ("tenant_max_tokens", 0),
    ("tenant_window_s", 0),
])
def test_invalid_tenant_budget_fails_before_startup(settings_values, name, value):
    with pytest.raises(ValidationError):
        Settings(**settings_values, **{name: value})


def test_prefix_load_slack_is_configurable_and_not_negative(settings_values, monkeypatch):
    monkeypatch.setenv("GATEWAY_PREFIX_LOAD_SLACK", "6")
    assert Settings(**settings_values).prefix_load_slack == 6

    with pytest.raises(ValidationError):
        Settings(**settings_values, prefix_load_slack=-1)


def test_batch_has_its_own_longer_queue_deadline_and_a_share_of_the_queue(settings_values, monkeypatch):
    defaults = Settings(**settings_values)

    assert defaults.queue_timeout_s == 5.0
    assert defaults.batch_queue_timeout_s == 30.0
    assert defaults.batch_queue_max_size == 8

    monkeypatch.setenv("GATEWAY_BATCH_QUEUE_TIMEOUT_S", "45")
    monkeypatch.setenv("GATEWAY_BATCH_QUEUE_MAX_SIZE", "4")
    configured = Settings(**settings_values)

    assert configured.batch_queue_timeout_s == 45.0
    assert configured.batch_queue_max_size == 4


@pytest.mark.parametrize("name", ["batch_queue_timeout_s", "batch_queue_max_size"])
def test_invalid_batch_queue_setting_fails_before_startup(settings_values, name):
    with pytest.raises(ValidationError):
        Settings(**settings_values, **{name: 0})
