"""Model Training API — port 8014."""

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
from services.model_training.service import meta as training_meta
from services.model_training.service import rollback, run_drift_check, run_training, versions

SERVICE = "model_training"
app = create_service_app(
    name=SERVICE,
    description=(
        "Model training: Walk Forward / Time Series Split, Bayesian HPO, Early Stopping, "
        "MLflow Tracking, Version Control, Automatic Retraining, Feature/Prediction/Concept Drift"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class TrainRequest(BaseModel):
    symbols: list[str] | str = Field(..., description="Ticker list or comma-separated")
    rebuild: bool = False
    n_trials: int = Field(15, ge=1, le=200)
    n_splits: int = Field(4, ge=2, le=12)
    validation_method: str = Field("walk_forward", description="walk_forward | time_series_split")
    use_mlflow: bool = True
    auto_retrain: bool = True
    force_retrain: bool = False


class DriftRequest(BaseModel):
    symbols: list[str] | str
    rebuild: bool = False
    n_trials: int = Field(3, ge=1, le=50)
    n_splits: int = Field(3, ge=2, le=8)


class RollbackRequest(BaseModel):
    name: str = "squeeze_probability"
    version: str


@router.get("/meta")
def meta() -> dict:
    return training_meta()


@router.post("/train", response_model=ServiceEnvelope)
def train(req: TrainRequest) -> ServiceEnvelope:
    data = run_training(
        req.symbols,
        rebuild=req.rebuild,
        n_trials=req.n_trials,
        n_splits=req.n_splits,
        validation_method=req.validation_method,
        use_mlflow=req.use_mlflow,
        auto_retrain=req.auto_retrain,
        force_retrain=req.force_retrain,
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


@router.post("/drift", response_model=ServiceEnvelope)
def drift(req: DriftRequest) -> ServiceEnvelope:
    data = run_drift_check(
        req.symbols,
        rebuild=req.rebuild,
        n_trials=req.n_trials,
        n_splits=req.n_splits,
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


@router.get("/versions")
def list_model_versions(name: str = "squeeze_probability") -> dict:
    return versions(name)


@router.post("/rollback", response_model=ServiceEnvelope)
def rollback_model(req: RollbackRequest) -> ServiceEnvelope:
    data = rollback(req.name, req.version)
    return ServiceEnvelope(
        service=SERVICE,
        symbol=None,
        as_of=None,
        degraded=not data.get("ok"),
        upstream_errors=[data.get("error", "")] if not data.get("ok") else [],
        data=data,
    )


app.include_router(router)
