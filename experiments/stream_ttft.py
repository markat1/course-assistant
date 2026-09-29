import json
import os
import time


import httpx

from gateway.monitoring.warmup_request import WARMUP_PREFIX

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://gateway:8780/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
RUNS = int(os.environ.get("RUNS", "6"))
QUESTIONS = [
    "Why does KV cache memory limit concurrency?",
    "What does prefix caching reuse?",
    "When is prefill the bottleneck?",
    "What does chunked prefill change?",
    "Why must a replica be warmed before traffic?",
    "What happens when the KV cache is full?",
]


def has_content(line: str) -> bool:
    if not line.startswith("data: ") or line == "data: [DONE]":
        return False
    choices = json.loads(line[6:]).get("choices") or []
    return bool(choices and (choices[0].get("delta") or {}).get("content"))


def measure(client: httpx.Client, question: str) -> tuple[float | None, float]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": WARMUP_PREFIX},
            {"role": "user", "content": question},
        ],
        "max_tokens": 64,
        "stream": True,
    }
    started = time.perf_counter()
    first = None
    with client.stream("POST", f"{GATEWAY_URL}/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if first is None and has_content(line):
                first = time.perf_counter() - started
    return first, time.perf_counter() - started


def main() -> None:
    print(f"gateway={GATEWAY_URL} model={MODEL} runs={RUNS}")
    with httpx.Client(timeout=60, trust_env=False) as client:
        for run in range(RUNS):
            first, total = measure(client, QUESTIONS[run % len(QUESTIONS)])
            print(f"run={run + 1} ttft_s={first:.3f} total_s={total:.3f}")


if __name__ == "__main__":
    main()