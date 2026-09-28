import os

import pytest
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def clear_app_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name)


def test_app_settings_default_to_local_gateway():
    from app.models.settings import AppSettings

    settings = AppSettings(_env_file=None)

    assert settings.model_name == "Qwen/Qwen3-8B"
    assert str(settings.gateway_url) == "http://127.0.0.1:8780/v1"
    assert settings.request_timeout_s == 30.0
    assert settings.max_turns == 6


def test_app_settings_use_app_environment_prefix(monkeypatch):
    from app.models.settings import AppSettings

    monkeypatch.setenv("APP_MODEL_NAME", "configured-model")
    monkeypatch.setenv("APP_GATEWAY_URL", "http://gateway:8780/v1")
    monkeypatch.setenv("APP_REQUEST_TIMEOUT_S", "45")
    monkeypatch.setenv("APP_MAX_TURNS", "4")
    monkeypatch.setenv("GATEWAY_MODEL_NAME", "unrelated-model")

    settings = AppSettings(_env_file=None)

    assert settings.model_name == "configured-model"
    assert str(settings.gateway_url) == "http://gateway:8780/v1"
    assert settings.request_timeout_s == 45.0
    assert settings.max_turns == 4


@pytest.mark.parametrize("values", [
    {"model_name": "   "},
    {"gateway_url": "not-a-url"},
    {"request_timeout_s": 0},
    {"request_timeout_s": float("inf")},
    {"max_turns": 0},
])
def test_app_settings_reject_invalid_runtime_configuration(values):
    from app.models.settings import AppSettings

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, **values)
