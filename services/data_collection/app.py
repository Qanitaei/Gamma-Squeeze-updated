"""Data Collection API — port 8001."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope, SymbolRequest
from services.data_collection.adapters_api import adapters_catalog, adapters_health, fetch_adapter
from services.data_collection.service import collect_snapshot

SERVICE = "data_collection"
app = create_service_app(
    name=SERVICE,
    description=(
        "Data Collection microservice (Phase 3): Historical Options Matrix "
        "(Alpaca API + alpaca-options-matrix-backup KV; IBKR/OPRA/Polygon/CBOE), "
        "Market Data, Macro, Structure, Corporate, Calendar adapters + SSD sync"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class AdapterFetchRequest(BaseModel):
    symbol: str | None = None
    as_of: str | None = None
    series_id: str | None = None
    series_ids: list[str] | None = None
    year: int | None = None
    lookback: int | None = None
    list_dates: bool = False
    extras: dict = Field(default_factory=dict)


@router.post("/collect", response_model=ServiceEnvelope)
def collect(req: SymbolRequest) -> ServiceEnvelope:
    data = collect_snapshot(req.symbol, as_of=req.as_of, lookback_days=req.lookback_days)
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("kv_dates_error") or data.get("ohlcv", {}).get("error")),
        upstream_errors=[e for e in [data.get("kv_dates_error"), data.get("ohlcv", {}).get("error")] if e],
        data=data,
    )


@router.get("/collect/{symbol}", response_model=ServiceEnvelope)
def collect_get(symbol: str, as_of: str | None = None, lookback_days: int = 60) -> ServiceEnvelope:
    return collect(SymbolRequest(symbol=symbol, as_of=as_of, lookback_days=lookback_days))


@router.get("/adapters")
def list_all_adapters() -> dict:
    return adapters_catalog()


@router.get("/adapters/health")
def health_all_adapters() -> dict:
    return adapters_health()


@router.post("/adapters/{adapter_name}/fetch", response_model=ServiceEnvelope)
def adapter_fetch(adapter_name: str, req: AdapterFetchRequest) -> ServiceEnvelope:
    try:
        kwargs = dict(req.extras)
        if req.year is not None:
            kwargs["year"] = req.year
        if req.lookback is not None:
            kwargs["lookback"] = req.lookback
        if req.series_id:
            kwargs["series_id"] = req.series_id
        if req.series_ids:
            kwargs["series_ids"] = req.series_ids
        if req.list_dates:
            kwargs["list_dates"] = True
        data = fetch_adapter(
            adapter_name,
            symbol=req.symbol,
            as_of=req.as_of,
            **kwargs,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ServiceEnvelope(
        service=SERVICE,
        symbol=(req.symbol or "").upper() or None,
        as_of=req.as_of or data.get("as_of"),
        degraded=not data.get("success", True),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/adapters/{adapter_name}/health")
def adapter_health(adapter_name: str) -> dict:
    from gamma_squeeze.data_sources.registry import get_adapter, load_all_adapters

    load_all_adapters()
    try:
        return get_adapter(adapter_name).health()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


app.include_router(router)
