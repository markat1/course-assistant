from time import monotonic
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

class WorkerState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(min_length=1, pattern=r"\S")
    base_url: AnyHttpUrl
    ready: bool = False

    engine_running: int | None = Field(default=None, ge=0)
    engine_waiting: int | None = Field(default=None, ge=0)
    kv_cache_usage_ratio: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )

    metrics_updated_at: float | None = Field(
        default=None,
        allow_inf_nan=False,
    )

    gateway_queue_depth: int = Field(default=0, ge=0)
    gateway_in_flight: int = Field(default=0, ge=0)
    ramp_limit: int | None = Field(default=None, ge=1)

    @property
    def metrics_age_s(self) -> float | None:
        """Return seconds since the last successful metrics update.

        Return None if no metrics have been collected. The update timestamp
        must use the same monotonic clock as this calculation.
        """
        if self.metrics_updated_at is None:
            return None
        
        return monotonic() - self.metrics_updated_at