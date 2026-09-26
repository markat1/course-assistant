from gateway.models.settings import Settings


from fastapi import FastAPI


from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for the FastAPI application."""
    app.state.settings = Settings()
    yield