"""Configuration primitives for the streaming data ML project."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import os


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""

    app_name: str
    kafka_bootstrap_servers: Tuple[str, ...]
    posts_raw_topic: str
    posts_enriched_topic: str
    posts_trends_topic: str
    rolling_window_seconds: int
    trend_zscore_threshold: float


def _get_env(name: str, default: str) -> str:
    """Read an environment variable with a fallback default."""
    return os.environ.get(name, default)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings with caching to avoid repeated environment parsing."""
    brokers = _get_env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    broker_tuple = tuple(item.strip() for item in brokers.split(",") if item.strip())

    return Settings(
        app_name=_get_env("STREAM_APP_NAME", "streaming_data_ml"),
        kafka_bootstrap_servers=broker_tuple,
        posts_raw_topic=_get_env("POSTS_RAW_TOPIC", "posts.raw"),
        posts_enriched_topic=_get_env("POSTS_ENRICHED_TOPIC", "posts.enriched"),
        posts_trends_topic=_get_env("POSTS_TRENDS_TOPIC", "posts.trends"),
        rolling_window_seconds=int(_get_env("ROLLING_WINDOW_SECONDS", "60")),
        trend_zscore_threshold=float(_get_env("TREND_ZSCORE_THRESHOLD", "3.0")),
    )

