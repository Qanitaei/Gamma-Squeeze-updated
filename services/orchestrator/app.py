"""API Gateway / Orchestrator — port 8000.

Exposes unified REST paths (/options, /features, /gex, …) plus legacy /v1/* routes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from services.common.app_factory import create_service_app
from services.common.http_client import get_json
from services.common.pipeline import CORE_PIPELINE_STEPS, PIPELINE_STEPS
from services.common.registry import SERVICE_REGISTRY, service_url
from services.common.schemas import ServiceEnvelope
from services.orchestrator.rest_api import REST_ENDPOINTS, router as rest_router
from services.orchestrator.service import pipeline_contract, run_pipeline

SERVICE = "orchestrator"
app = create_service_app(
    name=SERVICE,
    description=(
        "API gateway: unified REST (/options, /features, /gex, /gamma, /regime, "
        "/prediction, /tft, /xgboost, /ppo, /hedging, /dashboard, /alerts, "
        "/backtest, /train, /retrain) + ordered pipeline orchestrator "
        "(Data Collection → Trade Alert API)"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class PipelineRequest(BaseModel):
    symbol: str
    rebuild_features: bool = False
    mode: Literal["in_process", "http"] = Field(
        default="in_process",
        description="in_process calls domain modules; http fans out to peer /health APIs",
    )
    include_companions: bool = Field(
        default=True,
        description="Include explainability, model_training, performance_metrics stages",
    )


@router.get("/registry")
def registry() -> dict:
    return {
        "services": [
            {**info, "url": service_url(name)}
            for name, info in SERVICE_REGISTRY.items()
        ],
        "rest": list(REST_ENDPOINTS),
        "pipeline": pipeline_contract(),
    }


@router.get("/pipeline/contract")
def get_pipeline_contract() -> dict:
    """Locked DAG: Data Collection → Trade Alert API (+ companions)."""
    return pipeline_contract()


@router.get("/stack/inventory")
def stack_inventory(probe: bool = False) -> dict:
    """Locked Phase 2 programming stack (optionally probe local installs)."""
    from gamma_squeeze.stack.inventory import locked_stack, probe_stack

    return probe_stack() if probe else locked_stack()


@router.get("/standards")
def coding_standards() -> dict:
    """Locked Phase 19 coding standards catalog."""
    from gamma_squeeze.core.standards import standards_meta

    return standards_meta()


def _safe_health(fn, *, label: str) -> dict:
    try:
        out = fn()
        return out if isinstance(out, dict) else {label: out}
    except Exception as exc:  # noqa: BLE001
        return {label: "down", "error": str(exc)}


@router.get("/stack/health")
def stack_health() -> dict:
    from gamma_squeeze.stack.boosters import available_boosters
    from gamma_squeeze.stack.duckdb_store import health as duck_health
    from gamma_squeeze.stack.inventory import probe_stack
    from gamma_squeeze.stack.kafka_bus import health as kafka_health
    from gamma_squeeze.stack.mlflow_tracking import health as mlflow_health
    from gamma_squeeze.stack.polars_frames import health as polars_health
    from gamma_squeeze.stack.postgres import health as pg_health
    from gamma_squeeze.stack.redis_client import health as redis_health
    from gamma_squeeze.stack.settings import get_stack_settings

    settings = get_stack_settings()
    inventory = probe_stack()
    return {
        "phase": settings.phase,
        "python_requires": settings.python_requires,
        "python_ok": inventory["python_ok"],
        "settings": settings.to_public_dict(),
        "redis": _safe_health(redis_health, label="redis"),
        "postgres": _safe_health(pg_health, label="postgres"),
        "duckdb": _safe_health(duck_health, label="duckdb"),
        "kafka": _safe_health(kafka_health, label="kafka"),
        "mlflow": _safe_health(mlflow_health, label="mlflow"),
        "polars_arrow": _safe_health(polars_health, label="polars_arrow"),
        "boosters": available_boosters(),
        "core_ok": inventory["core_ok"],
        "core_missing": inventory["core_missing"],
        "inventory_summary": {
            cat: [c["name"] for c in comps]
            for cat, comps in inventory["categories"].items()
        },
    }


@router.get("/services/health")
def services_health() -> dict:
    statuses = {}
    for name in SERVICE_REGISTRY:
        if name == "orchestrator":
            statuses[name] = {"status": "ok", "url": service_url(name)}
            continue
        url = service_url(name) + "/health"
        try:
            statuses[name] = get_json(url, timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            statuses[name] = {"status": "down", "error": str(exc), "url": url}
    return {"services": statuses}


@router.get("/services/health/async")
async def services_health_async() -> dict:
    """Concurrent async health fan-out (preferred when many peers are up)."""
    from gamma_squeeze.core.async_http import gather_health

    urls = {
        name: service_url(name) + "/health"
        for name in SERVICE_REGISTRY
        if name != "orchestrator"
    }
    statuses = await gather_health(urls)
    statuses["orchestrator"] = {"status": "ok", "url": service_url("orchestrator")}
    return {"services": statuses, "mode": "async"}


@router.post("/pipeline/run", response_model=ServiceEnvelope)
def pipeline_run(req: PipelineRequest) -> ServiceEnvelope:
    data = run_pipeline(
        req.symbol,
        rebuild_features=req.rebuild_features,
        mode=req.mode,
        include_companions=req.include_companions,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=not data.get("ok", False),
        upstream_errors=[
            f"{t['step']}: {t['error']}" for t in data.get("stages", data.get("trace", [])) if t.get("error")
        ],
        data=data,
    )


@router.get("/pipeline/run")
def pipeline_run_get(
    symbol: str,
    rebuild_features: bool = False,
    mode: Literal["in_process", "http"] = Query(default="in_process"),
    include_companions: bool = Query(default=True),
) -> ServiceEnvelope:
    return pipeline_run(
        PipelineRequest(
            symbol=symbol,
            rebuild_features=rebuild_features,
            mode=mode,
            include_companions=include_companions,
        )
    )


app.include_router(router)
app.include_router(rest_router)


@app.get("/endpoints", tags=["rest"])
def list_endpoints() -> dict:
    return {
        "endpoints": list(REST_ENDPOINTS),
        "docs": "/docs",
        "openapi": "/openapi.json",
        "core_pipeline": list(CORE_PIPELINE_STEPS),
        "pipeline_steps": list(PIPELINE_STEPS),
    }
