"""Temporal Fusion Transformer API — port 8006."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel, Field

from gamma_squeeze.deep.temporal_model import (
    TFT_HORIZON_LABELS,
    TFT_HORIZONS,
    TFT_OUTPUTS,
    TFT_QUANTILES,
    TFT_TARGET_LABELS,
    TFT_TARGETS,
)
from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.tft_forecasting.service import run_tft

SERVICE = "tft_forecasting"
app = create_service_app(
    name=SERVICE,
    description=(
        "Temporal Fusion Transformer multi-target quantile forecasting: "
        "Future Price, IV, Net GEX, Dealer Hedge Requirement, Gamma Flip, "
        "Call/Put Wall, Expected Move — horizons 1/2/3/5/10 trading days with "
        "median, q10/q25/q75/q90, prediction intervals, and attention weights"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class TFTRequest(BaseModel):
    symbol: str
    train: bool = True
    lookback: int = Field(32, ge=8, le=128)
    horizons: list[int] | None = None
    targets: list[str] | None = None
    quantile_mode: str = Field("median_spread", pattern="^(full|median_spread)$")


@router.get("/horizons")
def list_horizons() -> dict:
    return {
        "horizons": list(TFT_HORIZONS),
        "labels": {str(h): TFT_HORIZON_LABELS[h] for h in TFT_HORIZONS},
        "targets": list(TFT_TARGETS),
        "target_labels": dict(TFT_TARGET_LABELS),
        "quantiles": list(TFT_QUANTILES),
        "outputs": list(TFT_OUTPUTS),
    }


@router.get("/meta")
def meta() -> dict:
    return {
        "service": SERVICE,
        "targets": list(TFT_TARGETS),
        "target_labels": dict(TFT_TARGET_LABELS),
        "horizons": list(TFT_HORIZONS),
        "horizon_labels": dict(TFT_HORIZON_LABELS),
        "quantiles": list(TFT_QUANTILES),
        "outputs": [
            "median_forecast",
            "q10",
            "q25",
            "q75",
            "q90",
            "prediction_interval",
            "attention_weights",
        ],
    }


@router.post("/forecast", response_model=ServiceEnvelope)
def forecast(req: TFTRequest) -> ServiceEnvelope:
    data = run_tft(
        req.symbol,
        train=req.train,
        lookback=req.lookback,
        horizons=req.horizons,
        targets=req.targets,
        quantile_mode=req.quantile_mode,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/forecast/{symbol}", response_model=ServiceEnvelope)
def forecast_get(
    symbol: str,
    train: bool = False,
    lookback: int = 32,
    quantile_mode: str = "full",
) -> ServiceEnvelope:
    return forecast(
        TFTRequest(
            symbol=symbol,
            train=train,
            lookback=lookback,
            quantile_mode=quantile_mode,
        )
    )


app.include_router(router)
