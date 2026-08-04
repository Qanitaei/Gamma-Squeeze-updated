"""Shared request/response models for microservices."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    version: str = "0.1.0"


class ReadyResponse(BaseModel):
    ready: bool
    service: str
    detail: str = ""


class SymbolRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=8)
    as_of: str | None = None
    lookback_days: int = 60


class ServiceEnvelope(BaseModel):
    success: bool = True
    service: str
    version: str = "0.1.0"
    symbol: str | None = None
    as_of: str | None = None
    degraded: bool = False
    upstream_errors: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
