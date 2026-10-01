import logging
import sys
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.api.chat import observe_engine_count
from gateway.execution import hops
from gateway.lifespan import create_token_counter
from gateway.models.chat.chat_request import ChatRequest
from gateway.models.settings import Settings
from gateway.monitoring.metrics import GATEWAY_REGISTRY
from gateway.monitoring.warmup_request import WARMUP_PREFIX
from gateway.policies.guard import inspect
from gateway.policies.hop_ledger import HopLedger
from gateway.policies.token_count import (
    build_token_counter,
    cache_by_digest,
    count_prompt_tokens,
    counter_name,
    estimate_tokens,
)

LOOKUP = {"type": "function", "function": {"name": "lookup_course", "parameters": {"type": "object"}}}
# usage.prompt_tokens from the engine, metrics/warmup-first-token-2026-09-29.txt
ENGINE_PROMPT_TOKENS = {
    "Why does KV cache memory limit concurrency?": 1673,
    "What does prefix caching reuse?": 1671,
    "When is prefill the bottleneck?": 1672,
    "What does chunked prefill change?": 1673,
    "Why must a replica be warmed before traffic?": 1674,
    "What happens when the KV cache is full?": 1674,
}


def words(text: str) -> int:
    """A fake tokenizer: one token per word."""
    return len(text.split())


def request(*messages: dict, **extra) -> ChatRequest:
    return ChatRequest.model_validate({"model": "Qwen/Qwen3-8B", "messages": list(messages), **extra})


def system(content: str) -> dict:
    return {"role": "system", "content": content}


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def settings(**values) -> Settings:
    return Settings(
        _env_file=None,
        model_name="Qwen/Qwen3-8B",
        worker_urls={"worker-a": "http://worker-a:8000/v1"},
        **values,
    )


def fake_tokenizers(monkeypatch, from_pretrained) -> None:
    module = SimpleNamespace(Tokenizer=SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setitem(sys.modules, "tokenizers", module)


def test_estimate_is_the_default_counter():
    assert settings().token_counter == "estimate"
    assert build_token_counter(settings()) is estimate_tokens
    assert count_prompt_tokens(request(user("Hello"))) == 5 // 4 + 1


def test_counter_is_read_from_the_gateway_environment(monkeypatch):
    monkeypatch.setenv("GATEWAY_TOKEN_COUNTER", "tokenizer")

    assert settings().token_counter == "tokenizer"


def test_estimate_mode_never_imports_the_tokenizer_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    assert build_token_counter(settings(token_counter="estimate")) is estimate_tokens


def test_tokenizer_mode_counts_with_the_served_models_tokenizer(monkeypatch):
    loaded = []

    def from_pretrained(model_name):
        loaded.append(model_name)
        return SimpleNamespace(encode=lambda text, add_special_tokens: SimpleNamespace(ids=text.split()))

    fake_tokenizers(monkeypatch, from_pretrained)

    counter = build_token_counter(settings(token_counter="tokenizer"))

    assert loaded == ["Qwen/Qwen3-8B"]
    assert counter("three short words") == 3
    assert counter_name(counter) == "tokenizer"


def test_gateway_falls_back_to_the_estimate_when_the_tokenizer_cannot_be_loaded(monkeypatch, caplog):
    def from_pretrained(model_name):
        raise OSError("no route to huggingface.co")

    fake_tokenizers(monkeypatch, from_pretrained)

    with caplog.at_level(logging.WARNING):
        counter = build_token_counter(settings(token_counter="tokenizer"))

    assert counter is estimate_tokens
    assert "could not be loaded" in caplog.text


def test_active_counter_is_exported_after_a_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    counter = create_token_counter(settings(token_counter="tokenizer"))

    assert counter is estimate_tokens
    assert GATEWAY_REGISTRY.get_sample_value("orch_token_counter_info", {"counter": "estimate"}) == 1
    assert GATEWAY_REGISTRY.get_sample_value("orch_token_counter_info", {"counter": "tokenizer"}) is None


def test_tokenizer_mode_adds_the_chat_template_to_the_message_contents():
    payload = request(system("You are the Course Tutor."), user("What is prefill?"))

    tokens = count_prompt_tokens(payload, words)

    # 5 + 3 words, 5 template tokens per message, 7 for the start of the reply
    assert tokens == 8 + 2 * 5 + 7


def test_tokenizer_mode_counts_the_tool_schema_that_the_estimate_ignores():
    plain = request(system("You are the Course Tutor."), user("What is prefill?"))
    with_tools = request(system("You are the Course Tutor."), user("What is prefill?"), tools=[LOOKUP])

    assert count_prompt_tokens(with_tools) == count_prompt_tokens(plain)
    assert count_prompt_tokens(with_tools, words) > count_prompt_tokens(plain, words) + 76


def test_tokenizer_mode_counts_tool_call_arguments_and_tool_results():
    question = [system("You are the Course Tutor."), user("What is prefill?")]
    tool_turn = [
        {"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "lookup_course", "arguments": '{"query": "prefill and decode"}'},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "Prefill processes the prompt in parallel."},
    ]

    before = count_prompt_tokens(request(*question), words)
    after = count_prompt_tokens(request(*question, *tool_turn), words)

    # two more messages, the call (7 words) and its wrapper, the result (6 words) and its wrapper
    assert after - before == 2 * 5 + (7 + 4) + (6 + 4)


def test_oversized_prompt_is_refused_without_being_tokenized():
    tokenized = []

    def counter(text):
        tokenized.append(text)
        return words(text)

    payload = request(user("a" * (16 * 8192 + 1)), max_tokens=16)

    guard = inspect(payload, context_length=8192, max_output_tokens=1024, counter=counter)

    assert (guard.status, guard.reason) == (400, "prompt_too_long")
    assert tokenized == []


def test_guard_reports_the_count_it_decided_on():
    payload = request(user("What is prefill?"), max_tokens=16)

    guard = inspect(payload, context_length=8192, max_output_tokens=1024, counter=words)

    assert guard.ok
    assert guard.prompt_tokens == 3 + 5 + 7


def test_prompt_the_estimate_refuses_can_fit_by_the_real_count():
    payload = request(user("x" * (4 * 7400)))

    by_estimate = inspect(payload, context_length=8192, max_output_tokens=1024)
    by_tokenizer = inspect(payload, context_length=8192, max_output_tokens=1024, counter=lambda text: 3700)

    assert by_estimate.reason == "prompt_too_long"
    assert by_tokenizer.ok


def test_repeated_text_is_tokenized_once():
    tokenized = []

    def counter(text):
        tokenized.append(text)
        return words(text)

    cached = cache_by_digest(counter)

    assert [cached("the shared prefix"), cached("the shared prefix"), cached("another")] == [3, 3, 1]
    assert tokenized == ["the shared prefix", "another"]


def test_cache_forgets_the_least_recently_used_text():
    tokenized = []

    def counter(text):
        tokenized.append(text)
        return words(text)

    cached = cache_by_digest(counter, max_entries=2)

    for text in ("prefix", "first", "prefix", "second", "prefix", "first"):
        cached(text)

    assert tokenized == ["prefix", "first", "second", "first"]


def test_hop_tokens_use_the_given_counter(monkeypatch):
    monkeypatch.setattr(hops, "HOP_LEDGER", HopLedger(max_prefixes=8))
    payload = request(system("You are the Course Tutor."), user("What is prefill?"))
    hops.record_placement(payload, "worker-a", words)

    hop = hops.record_placement(payload, "worker-b", words)

    assert hop is not None and hop.tokens == 5


def difference_samples(request_class: str, le: str) -> float:
    labels = {"request_class": request_class, "le": le}
    return GATEWAY_REGISTRY.get_sample_value("orch_prompt_token_difference_bucket", labels) or 0.0


def test_difference_from_the_engines_count_is_recorded_per_request_class():
    exact, all_before = difference_samples("batch", "0.0"), difference_samples("batch", "+Inf")
    response = JSONResponse({"usage": {"prompt_tokens": 1673}})

    observe_engine_count(response, 1675, "batch")

    assert difference_samples("batch", "+Inf") == all_before + 1
    assert difference_samples("batch", "2.0") - difference_samples("batch", "1.0") == 1
    assert difference_samples("batch", "0.0") == exact


@pytest.mark.parametrize("response", [
    JSONResponse({"error": "engine failed"}, status_code=500),
    JSONResponse({"choices": []}),
    StreamingResponse(iter([b"data: [DONE]\n\n"]), media_type="text/event-stream"),
])
def test_responses_without_an_engine_count_record_no_difference(response):
    before = difference_samples("interactive", "+Inf")

    observe_engine_count(response, 100, "interactive")

    assert difference_samples("interactive", "+Inf") == before


def test_tokenizer_count_equals_the_engines_prompt_tokens():
    """Needs Qwen's tokenizer.json (network or the Hugging Face cache); skipped without it."""
    counter = build_token_counter(settings(token_counter="tokenizer"))
    if counter is estimate_tokens:
        pytest.skip("Qwen tokenizer could not be loaded")

    counted = {
        question: count_prompt_tokens(request(system(WARMUP_PREFIX), user(question)), counter)
        for question in ENGINE_PROMPT_TOKENS
    }

    assert counted == ENGINE_PROMPT_TOKENS
