"""Structured logging for institutional ops."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "service"):
            payload["service"] = getattr(record, "service")
        if hasattr(record, "symbol"):
            payload["symbol"] = getattr(record, "symbol")
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_CONFIGURED = False


def setup_logging(*, level: str = "INFO", json_logs: bool = True) -> None:
    global _CONFIGURED
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _CONFIGURED = True


def get_logger(name: str = "gamma_squeeze") -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    msg: str,
    *,
    level: int = logging.INFO,
    service: str | None = None,
    symbol: str | None = None,
    **fields: Any,
) -> None:
    """Structured log helper with arbitrary fields."""
    extra: dict[str, Any] = {"extra_fields": fields}
    if service is not None:
        extra["service"] = service
    if symbol is not None:
        extra["symbol"] = symbol
    logger.log(level, msg, extra=extra)
