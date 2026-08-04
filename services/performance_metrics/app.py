"""Performance Metrics API — port 8015."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.performance_metrics.service import meta as metrics_meta_fn
from services.performance_metrics.service import track_arrays, track_symbols

SERVICE = "performance_metrics"
app = create_service_app(
    name=SERVICE,
    description=(
        "Performance metrics: Accuracy/Precision/Recall/F1/AUC/Brier/LogLoss, "
        "MAE/RMSE, Sharpe/Sortino/Calmar/DD/Return, Win Rate/Profit Factor, Tail Risk/ES"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class TrackRequest(BaseModel):
    symbols: list[str] | str = Field(..., description="Ticker list or comma-separated")
    rebuild: bool = False
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    alpha: float = Field(0.05, gt=0.0, lt=0.5)


class ArraysRequest(BaseModel):
    y_true: list[float] | None = None
    proba: list[float] | None = None
    y_pred: list[float] | None = None
    y_reg_true: list[float] | None = None
    y_reg_pred: list[float] | None = None
    returns: list[float] | None = None
    signal: list[float] | None = None
    trade_pnls: list[float] | None = None
    hold_periods: list[float] | None = None
    threshold: float = 0.5
    alpha: float = 0.05
    periods_per_year: float = 252.0
    symbols: list[str] = Field(default_factory=list)


@router.get("/meta")
def meta() -> dict:
    return metrics_meta_fn()


@router.post("/track", response_model=ServiceEnvelope)
def track(req: TrackRequest) -> ServiceEnvelope:
    data = track_symbols(
        req.symbols,
        rebuild=req.rebuild,
        threshold=req.threshold,
        alpha=req.alpha,
    )
    symbols = data.get("symbols") or []
    return ServiceEnvelope(
        service=SERVICE,
        symbol=symbols[0] if symbols else None,
        as_of=None,
        degraded=data.get("status") not in ("ok",),
        upstream_errors=[data["status"]] if data.get("status") not in ("ok",) else [],
        data=data,
    )


@router.get("/track/{symbol}", response_model=ServiceEnvelope)
def track_get(symbol: str, rebuild: bool = False, threshold: float = 0.5, alpha: float = 0.05) -> ServiceEnvelope:
    return track(TrackRequest(symbols=symbol, rebuild=rebuild, threshold=threshold, alpha=alpha))


@router.post("/compute", response_model=ServiceEnvelope)
def compute(req: ArraysRequest) -> ServiceEnvelope:
    data = track_arrays(req.model_dump())
    return ServiceEnvelope(
        service=SERVICE,
        symbol=(req.symbols[0] if req.symbols else None),
        as_of=None,
        degraded=data.get("status") not in ("ok",),
        upstream_errors=[data["status"]] if data.get("status") not in ("ok",) else [],
        data=data,
    )


app.include_router(router)
