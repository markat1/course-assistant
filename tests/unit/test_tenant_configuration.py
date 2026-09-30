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
