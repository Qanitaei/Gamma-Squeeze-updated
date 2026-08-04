"""Dealer Hedging Simulation API — port 8007."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel, Field

from gamma_squeeze.dealer.hedge_demand import (
    DEALER_ESTIMATE_LABELS,
    DEALER_ESTIMATES,
    DEALER_OUTPUT_LABELS,
    DEALER_OUTPUTS,
)
from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.dealer_hedging.service import run_dealer_hedge

SERVICE = "dealer_hedging"
app = create_service_app(
    name=SERVICE,
    description=(
        "Dealer hedging simulation — share purchases/sales, gamma ramp, strike migration, "
        "delta hedging, dynamic gamma, inventory, liquidity, exhaustion, position flip; "
        "outputs flow map, demand curve, expected hedging volume"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class DealerRequest(BaseModel):
    symbol: str
    as_of: str | None = None
    grid_pct: float = Field(0.08, ge=0.01, le=0.5)
    n_points: int = Field(17, ge=5, le=101)
    expected_move_pct: float | None = Field(None, description="Optional ± move band for hedging volume")


@router.get("/meta")
def meta() -> dict:
    return {
        "service": SERVICE,
        "estimates": list(DEALER_ESTIMATES),
        "estimate_labels": dict(DEALER_ESTIMATE_LABELS),
        "outputs": list(DEALER_OUTPUTS),
        "output_labels": dict(DEALER_OUTPUT_LABELS),
    }


@router.post("/hedge-demand", response_model=ServiceEnvelope)
def hedge_demand(req: DealerRequest) -> ServiceEnvelope:
    data = run_dealer_hedge(
        req.symbol,
        as_of=req.as_of,
        grid_pct=req.grid_pct,
        n_points=req.n_points,
        expected_move_pct=req.expected_move_pct,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/hedge-demand/{symbol}", response_model=ServiceEnvelope)
def hedge_demand_get(
    symbol: str,
    as_of: str | None = None,
    grid_pct: float = 0.08,
    n_points: int = 17,
    expected_move_pct: float | None = None,
) -> ServiceEnvelope:
    return hedge_demand(
        DealerRequest(
            symbol=symbol,
            as_of=as_of,
            grid_pct=grid_pct,
            n_points=n_points,
            expected_move_pct=expected_move_pct,
        )
    )


@router.post("/simulate", response_model=ServiceEnvelope)
def simulate(req: DealerRequest) -> ServiceEnvelope:
    return hedge_demand(req)


@router.get("/simulate/{symbol}", response_model=ServiceEnvelope)
def simulate_get(
    symbol: str,
    as_of: str | None = None,
    grid_pct: float = 0.08,
    n_points: int = 17,
    expected_move_pct: float | None = None,
) -> ServiceEnvelope:
    return hedge_demand_get(symbol, as_of, grid_pct, n_points, expected_move_pct)


app.include_router(router)
