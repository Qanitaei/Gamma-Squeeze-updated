"""Redis cache / pubsub adapter."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.stack.settings import get_stack_settings


def get_redis():
    """Return a redis.Redis client, or None if redis package/server unavailable."""
    try:
        import redis
    except ImportError:
        return None
    settings = get_stack_settings()
    try:
        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception:  # noqa: BLE001
        return None


def cache_set(key: str, value: str, *, ttl_s: int = 300) -> bool:
    client = get_redis()
    if client is None:
        return False
    client.setex(key, ttl_s, value)
    return True


def cache_get(key: str) -> str | None:
    client = get_redis()
    if client is None:
        return None
    return client.get(key)


def publish(channel: str, message: str) -> bool:
    client = get_redis()
    if client is None:
        return False
    client.publish(channel, message)
    return True


def health() -> dict[str, Any]:
    client = get_redis()
    return {
        "redis": "up" if client is not None else "down",
        "url": get_stack_settings().redis_url,
    }
