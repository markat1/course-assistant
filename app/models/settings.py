from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_name: str = Field(
        default="Qwen/Qwen3-8B",
        min_length=1,
        pattern=r"\S",
    )
    gateway_url: AnyHttpUrl = Field(
        default="http://127.0.0.1:8780/v1",
    )
    request_timeout_s: float = Field(
        default=30.0,
        gt=0,
        allow_inf_nan=False,
    )
    max_turns: int = Field(default=6, gt=0)

