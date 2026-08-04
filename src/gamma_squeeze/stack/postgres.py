"""PostgreSQL operational store (SQLAlchemy)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.stack.settings import get_stack_settings

_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    try:
        from sqlalchemy import create_engine
    except ImportError:
        return None
    settings = get_stack_settings()
    try:
        _ENGINE = create_engine(settings.database_url, pool_pre_ping=True)
        with _ENGINE.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return _ENGINE
    except Exception:  # noqa: BLE001
        _ENGINE = None
        return None


def health() -> dict[str, Any]:
    eng = get_engine()
    return {
        "postgres": "up" if eng is not None else "down",
        "url": get_stack_settings().database_url.split("@")[-1],
    }


def ensure_schema() -> bool:
    eng = get_engine()
    if eng is None:
        return False
    with eng.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS alerts (
              id BIGSERIAL PRIMARY KEY,
              symbol TEXT NOT NULL,
              as_of DATE,
              payload JSONB NOT NULL,
              created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS model_runs (
              id BIGSERIAL PRIMARY KEY,
              backend TEXT NOT NULL,
              version TEXT,
              metrics JSONB,
              created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
    return True
