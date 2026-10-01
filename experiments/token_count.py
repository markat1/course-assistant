"""Compare the gateway's prompt token count with Qwen's own chat template. No GPU.

uv run --with jinja2 python -m experiments.token_count
"""
import json
import os
import time
from types import SimpleNamespace

from huggingface_hub import hf_hub_download
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from experiments.first_token import QUESTIONS
from gateway.models.chat.chat_request import ChatRequest
from gateway.monitoring.warmup_request import WARMUP_PREFIX
from gateway.policies.token_count import build_token_counter, count_prompt_tokens, estimate_tokens

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
# usage.prompt_tokens for the same six prompts, metrics/warmup-first-token-2026-09-29.txt
ENGINE_PROMPT_TOKENS = [1673, 1671, 1672, 1673, 1674, 1674]

LOOKUP = {"type": "function", "function": {
    "name": "lookup_course",
    "description": "Search course notes using short English keywords",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Keywords describing the course concept to look up."}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}}
HANDOFF = {"type": "function", "function": {
    "name": "transfer_to_course_tutor",
    "description": "Handoff to the Course Tutor agent to handle the request.",
    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
}}
SYSTEM = {"role": "system", "content": "You are the Course Tutor. Use lookup_course before answering."}
USER = {"role": "user", "content": QUESTIONS[0]}
RESULT = '{"query":"KV cache concurrency","passages":[{"title":"KV memory","text":"Each sequence holds KV for every token."}]}'


def call(number: int, arguments: str) -> dict:
    return {"id": f"call-{number}", "type": "function", "function": {"name": "lookup_course", "arguments": arguments}}


def conversation(messages: int) -> list[dict]:
    turns = [{"role": "system", "content": "You are the Course Tutor."}]
    for turn in range(1, messages):
        role = "user" if turn % 2 else "assistant"
        turns.append({"role": role, "content": f"Turn {turn}: the KV cache stores keys and values."})
    return turns


SHAPES = [
    ("user only", [USER], None),
    ("system + user", [SYSTEM, USER], None),
    ("4 messages", conversation(4), None),
    ("8 messages", conversation(8), None),
    ("system + user, 1 tool", [SYSTEM, USER], [LOOKUP]),
    ("system + user, 2 tools", [SYSTEM, USER], [LOOKUP, HANDOFF]),
    ("user only, 1 tool", [USER], [LOOKUP]),
    ("1 tool call + result", [
        SYSTEM, USER,
        {"role": "assistant", "content": None, "tool_calls": [call(1, '{"query": "KV cache concurrency"}')]},
        {"role": "tool", "tool_call_id": "call-1", "content": RESULT},
    ], [LOOKUP]),
    ("2 tool calls + 2 results", [
        SYSTEM, USER,
        {"role": "assistant", "content": None,
         "tool_calls": [call(1, '{"query": "KV cache"}'), call(2, '{"query": "prefill"}')]},
        {"role": "tool", "tool_call_id": "call-1", "content": RESULT},
        {"role": "tool", "tool_call_id": "call-2", "content": '{"query":"prefill","passages":[]}'},
    ], [LOOKUP]),
    ("call, result, answer, next question", [
        SYSTEM, USER,
        {"role": "assistant", "content": "Let me look.", "tool_calls": [call(1, '{"query": "KV cache"}')]},
        {"role": "tool", "tool_call_id": "call-1", "content": RESULT},
        {"role": "assistant", "content": "KV memory grows with every token of every sequence."},
        {"role": "user", "content": "And prefill?"},
    ], [LOOKUP, HANDOFF]),
]


def load_template():
    """Qwen's chat template, rendered the way transformers renders it for the engine."""
    with open(hf_hub_download(MODEL, "tokenizer_config.json")) as config:
        source = json.load(config)["chat_template"]
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
    return environment.from_string(source)


def request(messages: list[dict], tools: list[dict] | None) -> ChatRequest:
    body = {"model": MODEL, "messages": messages}
    if tools:
        body["tools"] = tools
    return ChatRequest.model_validate(body)


def median_ms(counter, text: str, runs: int = 20) -> float:
    seconds = []
    for _ in range(runs):
        started = time.perf_counter()
        counter(text)
        seconds.append(time.perf_counter() - started)
    return sorted(seconds)[runs // 2] * 1000


def main() -> None:
    started = time.perf_counter()
    counter = build_token_counter(SimpleNamespace(token_counter="tokenizer", model_name=MODEL))
    print(f"model={MODEL} tokenizer_load_s={time.perf_counter() - started:.2f}")
    assert counter is not estimate_tokens, "the tokenizer could not be loaded"

    tokenizer = Tokenizer.from_pretrained(MODEL)
    template = load_template()

    def template_tokens(messages: list[dict], tools: list[dict] | None) -> int:
        prompt = template.render(messages=messages, tools=tools, add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer.encode(prompt, add_special_tokens=False).ids)

    def raw(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    print(f"\nprefix chars={len(WARMUP_PREFIX)} tokens={raw(WARMUP_PREFIX)} "
          f"chars_per_token={len(WARMUP_PREFIX) / raw(WARMUP_PREFIX):.2f}")

    print("\n# the six first_token.py prompts: engine vs template vs gateway")
    for question, engine in zip(QUESTIONS, ENGINE_PROMPT_TOKENS):
        messages = [{"role": "system", "content": WARMUP_PREFIX}, {"role": "user", "content": question}]
        payload = request(messages, None)
        print(f"engine={engine} template={template_tokens(messages, None)} "
              f"tokenizer={count_prompt_tokens(payload, counter)} "
              f"contents_only={raw(WARMUP_PREFIX) + raw(question)} "
              f"estimate={count_prompt_tokens(payload)}")

    print("\n# request shapes: template (what the engine tokenizes) vs gateway")
    for label, messages, tools in SHAPES:
        payload = request(messages, tools)
        rendered = template_tokens(messages, tools)
        counted = count_prompt_tokens(payload, counter)
        print(f"{label:36s} template={rendered:4d} tokenizer={counted:4d} difference={counted - rendered:+d} "
              f"estimate={count_prompt_tokens(payload):4d} difference={count_prompt_tokens(payload) - rendered:+d}")

    long_prompt = WARMUP_PREFIX * 5
    print("\n# tokenizing time, median of 20, uncached")
    print(f"prefix tokens={raw(WARMUP_PREFIX)} ms={median_ms(raw, WARMUP_PREFIX):.2f}")
    print(f"long tokens={raw(long_prompt)} ms={median_ms(raw, long_prompt):.2f}")
    print(f"prefix through the gateway counter (cached) ms={median_ms(counter, WARMUP_PREFIX):.3f}")


if __name__ == "__main__":
    main()
