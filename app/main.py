from agents import Model
from fastapi import FastAPI

from app.api.routes import router
from app.lifespan import lifespan

def create_app(model: Model,*, max_turns: int = 6) -> FastAPI:
    app = FastAPI(title="Course Assistant")
    app.state.model = model
    app.state.max_turns = max_turns
    app.include_router(router)
    return app

def create_runtime_app() -> FastAPI:
    app = FastAPI(
        title="Course Assistant",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app