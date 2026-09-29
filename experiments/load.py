import asyncio
import collections
import os
import statistics
import time

import httpx

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://gateway:8780/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
REQUESTS = int(os.environ.get("REQUESTS", "100"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))

PREFIX = "You are the Course Tutor.\n" + (
    "Course passage: A KV cache stores attention keys and values for every "
    "previous token so decoding does not recompute them. "
) * 30


def build_payload(index: int) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PREFIX},
            {"role": "user", "content": f"Question {index}: explain one serving trade-off."},
        ],
        "max_tokens": MAX_TOKENS,
    }


async def send(
    client: httpx.AsyncClient,
    index: int,
    limit: asyncio.Semaphore,
    outcomes: collections.Counter,
    latencies: list[float],
) -> None:
    async with limit:
        started = time.perf_counter()
        try:
            response = await client.post(f"{GATEWAY_URL}/chat/completions", json=build_payload(index))
        except httpx.HTTPError as exc:
            outcomes[type(exc).__name__] += 1
            return
        outcomes[response.status_code] += 1
        if response.status_code == 200:
            latencies.append(time.perf_counter() - started)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


async def main() -> None:
    outcomes: collections.Counter = collections.Counter()
    latencies: list[float] = []
    limit = asyncio.Semaphore(CONCURRENCY)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        await asyncio.gather(*(send(client, i, limit, outcomes, latencies) for i in range(REQUESTS)))
    elapsed = time.perf_counter() - started
    print(f"concurrency={CONCURRENCY} requests={REQUESTS} max_tokens={MAX_TOKENS} seconds={elapsed:.1f}")
    print(f"outcomes={dict(outcomes)}")
    if latencies:
        print(
            f"ok_latency_s p50={statistics.median(latencies):.2f} "
            f"p99={percentile(latencies, 0.99):.2f} max={max(latencies):.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
