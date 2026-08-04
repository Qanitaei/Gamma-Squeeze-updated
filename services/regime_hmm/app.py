"""Market Regime HMM API — port 8004."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel

from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.regime_hmm.service import run_regime
from gamma_squeeze.regime.hmm_regime import HMM_OUTPUT_FIELDS, N_STATES, REGIME_STATES

SERVICE = "regime_hmm"
app = create_service_app(
    name=SERVICE,
    description=(
        "Hidden Markov Model market regime classification — 16 named states "
        "(Bull…Panic); outputs current_state, transition_matrix, "
        "next_state_probability, confidence"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class RegimeRequest(BaseModel):
    symbol: str
    train: bool = True
    include_macro: bool = True
    include_technicals: bool = True


@router.get("/states")
def list_states() -> dict:
    return {
        "states": list(REGIME_STATES),
        "n_states": N_STATES,
        "outputs": list(HMM_OUTPUT_FIELDS),
    }


@router.get("/meta")
def meta() -> dict:
    return {
        "service": SERVICE,
        "hidden_states": list(REGIME_STATES),
        "n_states": N_STATES,
        "outputs": list(HMM_OUTPUT_FIELDS),
        "method": "rule-informed emissions + sticky transition matrix + forward filter",
    }


@router.post("/regime", response_model=ServiceEnvelope)
def regime(req: RegimeRequest) -> ServiceEnvelope:
    data = run_regime(
        req.symbol,
        train=req.train,
        include_macro=req.include_macro,
        include_technicals=req.include_technicals,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")) or not data.get("current_state"),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/regime/{symbol}", response_model=ServiceEnvelope)
def regime_get(
    symbol: str,
    train: bool = True,
    include_macro: bool = True,
    include_technicals: bool = True,
) -> ServiceEnvelope:
    return regime(
        RegimeRequest(
            symbol=symbol,
            train=train,
            include_macro=include_macro,
            include_technicals=include_technicals,
        )
    )


app.include_router(router)
