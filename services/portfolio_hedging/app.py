"""Portfolio Hedging Engine API — port 8010."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel, Field

from gamma_squeeze.risk.portfolio_hedging_engine import OPTIMIZE_OBJECTIVES, engine_meta
from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.portfolio_hedging.service import run_portfolio_hedge

SERVICE = "portfolio_hedging"
app = create_service_app(
    name=SERVICE,
    description=(
        "Portfolio hedging engine — recommend Long Calls/Puts, Covered/Protective, "
        "spreads, Iron Condor, Collar, Dynamic Delta/Gamma/Vega/Theta hedges; "
        "optimize Return, Risk, Capital Efficiency, Margin"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class PortfolioRequest(BaseModel):
    symbol: str
    optimize_for: str = Field(
        "Risk",
        description="Primary objective: Return | Risk | Capital Efficiency | Margin",
    )
    use_live_alpaca: bool = True
    use_composite_forecast: bool = True
    portfolio: dict[str, float] | None = None


@router.get("/meta")
def meta() -> dict:
    return engine_meta()


@router.get("/structures")
def structures() -> dict:
    m = engine_meta()
    return {"recommend": m["recommend"], "optimize": m["optimize"]}


@router.post("/hedges", response_model=ServiceEnvelope)
def hedges(req: PortfolioRequest) -> ServiceEnvelope:
    obj = req.optimize_for if req.optimize_for in OPTIMIZE_OBJECTIVES else "Risk"
    data = run_portfolio_hedge(
        req.symbol,
        optimize_for=obj,
        use_live_alpaca=req.use_live_alpaca,
        use_composite_forecast=req.use_composite_forecast,
        portfolio=req.portfolio,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/hedges/{symbol}", response_model=ServiceEnvelope)
def hedges_get(
    symbol: str,
    optimize_for: str = "Risk",
    use_live_alpaca: bool = True,
    use_composite_forecast: bool = True,
) -> ServiceEnvelope:
    return hedges(
        PortfolioRequest(
            symbol=symbol,
            optimize_for=optimize_for,
            use_live_alpaca=use_live_alpaca,
            use_composite_forecast=use_composite_forecast,
        )
    )


@router.post("/optimize", response_model=ServiceEnvelope)
def optimize(req: PortfolioRequest) -> ServiceEnvelope:
    return hedges(req)


app.include_router(router)
