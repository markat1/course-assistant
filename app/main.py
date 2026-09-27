from agents import Model
from fastapi import FastAPI

from app.api.routes import router

def create_app(model: Model,*, max_turns: int = 6) -> FastAPI:
    app = FastAPI(title="Course Assistant")
    app.state.model = model
    app.state.max_turns = max_turns
    app.include_router(router)
    return app