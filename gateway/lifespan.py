import httpx
from fastapi import FastAPI

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from gateway.models.settings import Settings

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for the FastAPI application."""
    settings = Settings()
    app.state.settings = settings

    async with httpx.AsyncClient(timeout=settings.upstream_timeout_s, trust_env=False) as client:
        app.state.client = client
        yield