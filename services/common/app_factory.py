"""FastAPI app factory for independent microservices."""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import FastAPI, Request, Response

from services.common.schemas import HealthResponse, ReadyResponse


def create_service_app(
    *,
    name: str,
    description: str,
    version: str = "0.1.0",
    ready_fn: Callable[[], tuple[bool, str]] | None = None,
) -> FastAPI:
    """
    Build a microservice app with /health, /ready, structured request metrics.

    Domain logic stays in `service.py`; this factory only provides the HTTP shell.
    """
    app = FastAPI(title=name, description=description, version=version)
    app.state.service_name = name
    app.state.service_version = version

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            try:
                from gamma_squeeze.core.logging import get_logger, log_event
                from gamma_squeeze.core.metrics_collector import get_metrics

                metrics = get_metrics()
                metrics.incr("http.requests")
                metrics.timing("http.latency_ms", elapsed_ms)
                status = getattr(response, "status_code", 0) if response is not None else 0
                log_event(
                    get_logger(name),
                    "http_request",
                    service=name,
                    path=str(request.url.path),
                    method=request.method,
                    status=status,
                    latency_ms=round(elapsed_ms, 2),
                )
            except Exception:  # noqa: BLE001
                pass

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(service=name, version=version)

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready() -> ReadyResponse:
        if ready_fn is None:
            return ReadyResponse(ready=True, service=name, detail="ok")
        ok, detail = ready_fn()
        return ReadyResponse(ready=bool(ok), service=name, detail=str(detail))

    @app.get("/metrics", tags=["ops"])
    async def metrics() -> dict[str, Any]:
        try:
            from gamma_squeeze.core.metrics_collector import get_metrics

            return get_metrics().snapshot()
        except Exception as exc:  # noqa: BLE001
            return {"enabled": False, "error": str(exc)}

    @app.get("/", tags=["ops"])
    async def root() -> dict[str, Any]:
        return {
            "service": name,
            "version": version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
        }

    return app
