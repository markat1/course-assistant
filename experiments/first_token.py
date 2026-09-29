import os
import time

import httpx

from gateway.monitoring.warmup_request import WARMUP_PREFIX

TARGET_URL = os.environ.get("TARGET_URL", "http://host.docker.internal:30002/v1")
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

def shared_prefix() -> str:
    return WARMUP_PREFIX

def build_payload(question:str) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": shared_prefix()},
            {"role": "user", "content": question},
        ],
        "max_tokens":1,
    }

def measure(client: httpx.Client, question: str) -> tuple[float, int]:
    started = time.perf_counter()
    response = client.post(f"{TARGET_URL}/chat/completions", json=build_payload(question))
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return elapsed, response.json()["usage"]["prompt_tokens"]

def main() -> None:
    print(f"target={TARGET_URL} model={MODEL} runs={RUNS}")
    with httpx.Client(timeout=60, trust_env=False) as client:
        for run in range(RUNS):
            question = QUESTIONS[run % len(QUESTIONS)]
            elapsed, prompt_tokens = measure(client, question)
            print(f"run{run + 1} seconds={elapsed:.3f} prompt_tokens={prompt_tokens}")

if __name__ == "__main__":
    main()