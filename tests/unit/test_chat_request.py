import pytest
from pydantic import ValidationError

from gateway.models.chat.chat_request import ChatRequest


@pytest.fixture
def request_body():
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }


@pytest.mark.parametrize("max_tokens", [1, 48])
def test_positive_token_limit_is_preserved(request_body, max_tokens):
    payload = ChatRequest.model_validate({**request_body, "max_tokens": max_tokens})

    assert payload.max_tokens == max_tokens
    assert payload.model_dump(mode="json", exclude_unset=True)["max_tokens"] == max_tokens


@pytest.mark.parametrize("max_tokens", [0, -1, "48", "Explain prefill", True, 1.5])
def test_invalid_token_limit_is_rejected(request_body, max_tokens):
    with pytest.raises(ValidationError) as caught:
        ChatRequest.model_validate({**request_body, "max_tokens": max_tokens})

    assert any(error["loc"] == ("max_tokens",) for error in caught.value.errors())


def test_omitted_token_limit_stays_omitted(request_body):
    payload = ChatRequest.model_validate(request_body)

    assert payload.model_dump(mode="json", exclude_unset=True) == request_body


def test_explicit_null_token_limit_is_preserved(request_body):
    body = {**request_body, "max_tokens": None}
    payload = ChatRequest.model_validate(body)

    assert payload.model_dump(mode="json", exclude_unset=True) == body


def test_engine_extension_is_preserved(request_body):
    body = {**request_body, "max_tokens": 48, "top_k": 20}
    payload = ChatRequest.model_validate(body)

    assert payload.model_dump(mode="json", exclude_unset=True) == body
