from pydantic import BaseModel, Field

class WorkerMetrics(BaseModel):
    engine_running: int = Field(ge=0)
    engine_waiting: int = Field(ge=0)
    kv_cache_usage_ratio: float = Field(
        ge=0,
        le=1,
        allow_inf_nan=False,
    )