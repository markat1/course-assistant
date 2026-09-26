from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    model_name: str = Field(min_length=1, pattern=r"\S")
    worker_urls: dict[str, AnyHttpUrl] = Field(min_length=1)
    upstream_timeout_s: float = Field(
        default=20.0,
        gt=0,
        allow_inf_nan=False
    )
    metrics_interval_s: float = Field(
        default=5.0,
        gt=0,
        allow_inf_nan=False
    )
    metrics_max_age_s: float = Field(
        default=10.0,
        gt=0,
        allow_inf_nan=False
    )
    kv_usage_limit: float = Field(
        default=0.90,
        gt=0,
        lt=1,
        allow_inf_nan=False
    )