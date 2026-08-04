"""Kafka / Redpanda producer adapter (optional)."""

from __future__ import annotations

import json
from typing import Any

from gamma_squeeze.stack.settings import get_stack_settings

TOPICS = {
    "features": "gamma.features",
    "forecasts": "gamma.forecasts",
    "alerts": "gamma.alerts",
    "actions": "gamma.actions",
}


def get_producer():
    try:
        from confluent_kafka import Producer
    except ImportError:
        return None
    settings = get_stack_settings()
    try:
        return Producer({"bootstrap.servers": settings.kafka_bootstrap})
    except Exception:  # noqa: BLE001
        return None


def publish_json(topic_key: str, payload: dict[str, Any], *, key: str | None = None) -> bool:
    producer = get_producer()
    if producer is None:
        return False
    topic = TOPICS.get(topic_key, topic_key)
    producer.produce(topic, json.dumps(payload).encode("utf-8"), key=key.encode() if key else None)
    producer.flush(5)
    return True


def health() -> dict[str, Any]:
    return {
        "kafka_bootstrap": get_stack_settings().kafka_bootstrap,
        "producer": "configured" if get_producer() is not None else "unavailable",
        "topics": TOPICS,
    }
