"""XGBoost Direction API — port 8005."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel, Field

from gamma_squeeze.models.xgboost_direction import (
    DEFAULT_BACKEND,
    DIRECTION_HORIZON_LABELS,
    DIRECTION_HORIZONS,
    DIRECTION_TARGETS,
    DIRECTION_TRAINING,
    HAS_XGB,
)
from gamma_squeeze.stack.boosters import available_boosters
from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.xgboost_direction.service import run_direction

SERVICE = "xgboost_direction"
app = create_service_app(
    name=SERVICE,
    description=(
        "XGBoost multi-horizon direction (1/3/5/10/20d): classification, "
        "regression, probability — trained with TimeSeries CV, Optuna Bayesian "
        "optimization, and SHAP explainability"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class DirectionRequest(BaseModel):
    symbol: str
    train: bool = True
    horizon: int | None = None
    horizons: list[int] | None = None
    n_trials: int = Field(15, ge=0, le=100)
    cv_folds: int = Field(5, ge=2, le=10)
    include_shap: bool = True
    direction_threshold: float = 0.005


@router.get("/horizons")
def list_horizons() -> dict:
    return {
        "horizons": list(DIRECTION_HORIZONS),
        "labels": {str(h): DIRECTION_HORIZON_LABELS[h] for h in DIRECTION_HORIZONS},
        "targets": list(DIRECTION_TARGETS),
        "training": list(DIRECTION_TRAINING),
    }


@router.get("/meta")
def meta() -> dict:
    return {
        "service": SERVICE,
        "horizons": list(DIRECTION_HORIZONS),
        "horizon_labels": dict(DIRECTION_HORIZON_LABELS),
        "targets": list(DIRECTION_TARGETS),
        "training": list(DIRECTION_TRAINING),
        "backend_default": DEFAULT_BACKEND,
        "xgboost_available": HAS_XGB,
        "boosters": available_boosters(),
    }


@router.post("/direction", response_model=ServiceEnvelope)
def direction(req: DirectionRequest) -> ServiceEnvelope:
    data = run_direction(
        req.symbol,
        train=req.train,
        horizon=req.horizon,
        horizons=req.horizons,
        n_trials=req.n_trials,
        cv_folds=req.cv_folds,
        include_shap=req.include_shap,
        direction_threshold=req.direction_threshold,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/direction/{symbol}", response_model=ServiceEnvelope)
def direction_get(
    symbol: str,
    train: bool = False,
    horizon: int | None = None,
    n_trials: int = 15,
    cv_folds: int = 5,
    include_shap: bool = True,
) -> ServiceEnvelope:
    return direction(
        DirectionRequest(
            symbol=symbol,
            train=train,
            horizon=horizon,
            n_trials=n_trials,
            cv_folds=cv_folds,
            include_shap=include_shap,
        )
    )


app.include_router(router)
