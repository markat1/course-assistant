from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"

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
    queue_max_size: int = Field(default=16, gt=0)
    dispatch_concurrency_per_worker: int = Field(default=8, gt=0)
    queue_timeout_s: float = Field(
        default=5.0,
        gt=0,
        allow_inf_nan=False
    )
    context_length: int = Field(default=8192, gt=0)
    max_output_tokens: int = Field(default=1024, gt=0)
    capacity_retry_after_s: int = Field(default=2, gt=0)
    warmup_prompt: str = Field(
        default=(
            "Course context: A KV cache stores attention keys and values "
            "from previous tokens for reuse during generation. "
            "Question: Why does KV-cache memory grow during a conversation?"
        ),
        min_length=1,
        pattern=r"\S",
    )
    warmup_max_tokens: int = Field(default=128, gt=0)
    warmup_timeout_s: float = Field(
        default=20.0,
        gt=0,
        allow_inf_nan=False,
    )
    tenant_max_tokens: int = Field(default=200_000, gt=0)
    tenant_window_s: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    prefix_load_slack: int = Field(default=4, ge=0)
    token_counter: Literal["estimate", "tokenizer"] = "estimate"