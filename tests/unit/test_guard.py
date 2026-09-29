from gateway.models.chat.chat_request import ChatRequest
from gateway.policies.guard import inspect


def request(content: str, max_tokens: int | None = None) -> ChatRequest:
    body = {"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": content}]}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return ChatRequest.model_validate(body)


def test_ordinary_request_passes_the_guard():
    guard = inspect(request("Explain the KV cache.", 128), context_length=8192, max_output_tokens=1024)

    assert guard.ok
    assert guard.status == 200


def test_output_budget_above_the_limit_is_rejected():
    guard = inspect(request("Hello", 2048), context_length=8192, max_output_tokens=1024)

    assert not guard.ok
    assert (guard.status, guard.reason) == (400, "bad_max_tokens")


def test_prompt_that_cannot_fit_the_context_is_rejected():
    guard = inspect(request("word " * 8000, 256), context_length=8192, max_output_tokens=1024)

    assert not guard.ok
    assert (guard.status, guard.reason) == (400, "prompt_too_long")


def test_missing_output_budget_counts_as_the_limit_when_checking_length():
    near_limit = "x" * (4 * 7400)

    assert inspect(request(near_limit, 64), context_length=8192, max_output_tokens=1024).ok
    assert inspect(request(near_limit), context_length=8192, max_output_tokens=1024).reason == "prompt_too_long"
