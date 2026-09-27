import asyncio

import httpx

from gateway.models.settings import Settings
from gateway.models.workers.worker_state import WorkerState
from gateway.monitoring.readiness import prepare_worker
from gateway.monitoring.warmup_request import build_warmup_request

async def check_worker(
        client: httpx.AsyncClient,
        worker: WorkerState,
        settings: Settings,
) -> None:
    print(f"Checking {worker.id}: {worker.base_url}", flush=True)

    await prepare_worker(
        client,
        worker,
        build_warmup_request(settings),
        warmup_timeout_s=settings.warmup_timeout_s,
        max_metrics_age_s=settings.metrics_max_age_s
    )

    print(f"PASS {worker.id}: health, model, completion and fresh metrics")
    print(worker.model_dump_json(indent=2))

async def main() -> None:
    settings = Settings()
    if len(settings.worker_urls) < 2:
        raise ValueError("Smoke test requires at least two configured workers.")

    async with httpx.AsyncClient(
        timeout=settings.upstream_timeout_s,
        trust_env=False,
    ) as client:
        for worker_id, base_url in settings.worker_urls.items():
            worker = WorkerState(id=worker_id, base_url=base_url)
            await check_worker(client, worker, settings)

if __name__ == "__main__":
    asyncio.run(main())