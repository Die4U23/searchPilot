"""运行配置：环境变量前缀 SEARCHPILOT_，字段与默认值见 docs/dev/build-plan.md 第 5 节。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """服务配置。未设置的可选项为 None。"""

    model_config = SettingsConfigDict(env_prefix="SEARCHPILOT_", extra="ignore")

    database_url: str | None = None
    data_dir: Path = Path("./data")
    artifact_dir: Path = Path("./artifacts")
    data_version: str | None = None
    max_query_chars: int = Field(default=256, ge=1)
    max_limit: int = Field(default=100, ge=1)
    log_level: str = "INFO"
    db_pool_min: int = Field(default=1, ge=0)
    db_pool_max: int = Field(default=4, ge=1)

    @field_validator("database_url", "data_version", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """空字符串（如 compose 里未赋值的变量）视为未设置。"""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        return value.strip().upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例；测试通过 `clear_settings_cache()` 重新读取环境变量。"""
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
