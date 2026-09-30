from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from openai import AsyncOpenAI

from app.llm import create_model
from app.models.settings import AppSettings

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = AppSettings()
    app.state.settings = settings
    app.state.max_turns = settings.max_turns

    async with httpx.AsyncClient(trust_env=False) as http_client:
        async with AsyncOpenAI(
            base_url=str(settings.gateway_url),
            api_key="local",
            max_retries=0,
            timeout=settings.request_timeout_s,
            http_client=http_client,
            default_headers={
                "X-Request-Class": "interactive",
                "X-Tenant": settings.tenant,
            },
        ) as client:
            app.state.model = create_model(
                model_name=settings.model_name,
                client=client,
            )

            try:
                yield
            finally:
                app.state.model = None