"""Gamma Squeeze Probability Engine API — port 8008."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel

from gamma_squeeze.config import alpaca_configured, alpaca_credentials
from gamma_squeeze.models.composite_squeeze import (
    COMPOSITE_INPUT_LABELS,
    COMPOSITE_INPUTS,
    COMPOSITE_OUTPUT_LABELS,
    COMPOSITE_OUTPUTS,
    COMPOSITE_SCORE_LABELS,
    COMPOSITE_SCORES,
)
from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.gamma_squeeze_engine.service import run_squeeze

SERVICE = "gamma_squeeze_engine"
app = create_service_app(
    name=SERVICE,
    description=(
        "Composite gamma squeeze engine — blends Hidden Markov, XGBoost, Temporal "
        "Transformer, Dealer Simulation, Pattern Recognition, Macro, Sentiment into "
        "scores and final probability / magnitude / duration / start / confidence / risk"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class SqueezeRequest(BaseModel):
    symbol: str
    persist: bool = True
    composite: bool = True


@router.get("/meta")
def meta() -> dict:
    creds = alpaca_credentials()
    return {
        "service": SERVICE,
        "inputs": list(COMPOSITE_INPUTS),
        "input_labels": dict(COMPOSITE_INPUT_LABELS),
        "scores": list(COMPOSITE_SCORES),
        "score_labels": dict(COMPOSITE_SCORE_LABELS),
        "outputs": list(COMPOSITE_OUTPUTS),
        "output_labels": dict(COMPOSITE_OUTPUT_LABELS),
        "alpaca": {
            "configured": alpaca_configured(),
            "trading_base": creds.get("trading_base"),
            "paper": creds.get("paper"),
            "key_present": bool(creds.get("key_id")),
        },
    }


@router.post("/squeeze", response_model=ServiceEnvelope)
def squeeze(req: SqueezeRequest) -> ServiceEnvelope:
    data = run_squeeze(req.symbol, persist=req.persist, composite=req.composite)
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error") or data.get("validation_errors")),
        upstream_errors=([data["error"]] if data.get("error") else [])
        + (data.get("validation_errors") or []),
        data=data,
    )


@router.get("/squeeze/{symbol}", response_model=ServiceEnvelope)
def squeeze_get(symbol: str, persist: bool = True, composite: bool = True) -> ServiceEnvelope:
    return squeeze(SqueezeRequest(symbol=symbol, persist=persist, composite=composite))


app.include_router(router)
