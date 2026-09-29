import pytest
from pydantic import ValidationError

from gateway.models.settings import Settings


@pytest.fixture
def settings_values():
    return {
        "_env_file": None,
        "model_name": "configured-model",
        "worker_urls": {"worker-a": "http://worker-a:8000/v1"},
    }


def test_configured_warmup_builds_a_bounded_request(settings_values):
    from gateway.monitoring.warmup_request import build_warmup_request

    settings = Settings(
        **settings_values,
        warmup_prompt="Explain the difference between prefill and decode.",
        warmup_max_tokens=48,
        warmup_timeout_s=7.0,
    )

    payload = build_warmup_request(settings)
    body = payload.model_dump(mode="json", exclude_unset=True)

    assert payload.model == "configured-model"
    assert payload.messages[-1].role == "user"
    assert payload.messages[-1].content == settings.warmup_prompt
    assert body["max_tokens"] == 48
    assert body["stream"] is False
    assert settings.warmup_timeout_s == 7.0


@pytest.mark.parametrize("name,value", [
    ("warmup_prompt", "   "),
    ("warmup_max_tokens", 0),
    ("warmup_timeout_s", 0),
])
def test_invalid_warmup_settings_fail_before_worker_startup(settings_values, name, value):
    with pytest.raises(ValidationError):
        Settings(**settings_values, **{name: value})


def test_warmup_requests_share_one_prefix_with_different_questions(settings_values):
    from gateway.monitoring.warmup_request import WARMUP_PREFIX, build_warmup_requests

    settings = Settings(**settings_values, warmup_max_tokens=16)

    payloads = build_warmup_requests(settings)
    questions = [payload.messages[-1].content for payload in payloads]

    assert len(payloads) == 5
    assert [payload.messages[0].content for payload in payloads] == [WARMUP_PREFIX] * 5
    assert questions[0] == settings.warmup_prompt
    assert len(set(questions)) == 5
    assert all(
        payload.model_dump(mode="json", exclude_unset=True)["max_tokens"] == 16
        for payload in payloads
    )
