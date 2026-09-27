import asyncio
import httpx

from gateway.models.workers.model_list import ModelList
from gateway.models.workers.worker_state import WorkerState

async def check_worker_model(
        client: httpx.AsyncClient,
        worker: WorkerState,
        *,
        model_name: str,
) -> None:
    url = f"{str(worker.base_url).rstrip('/')}/models"

    async with asyncio.timeout(2.0):
        response = await client.get(url)
        response.raise_for_status()

    models = ModelList.model_validate_json(response.content)

    if not models.contains(model_name):
        raise ValueError(f"Configured model is not served: {model_name}")
