"""Visualization Dashboard API — port 8011 (JSON panels; Alpaca-forward, no HTML desk)."""

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
from services.dashboard.service import build_dashboard, meta as dashboard_meta

SERVICE = "dashboard"
app = create_service_app(
    name=SERVICE,
    description="Institutional visualization dashboard — JSON panels + heatmaps (Alpaca)",
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class DashRequest(BaseModel):
    symbol: str
    use_live_alpaca: bool = True
    full: bool = Field(False, description="Enrich with regime/patterns/hedges/alerts/dealer calls")


@router.get("/meta")
def meta() -> dict:
    """Alpaca API health + panel catalog."""
    return dashboard_meta()


@router.post("/dashboard", response_model=ServiceEnvelope)
def dashboard(req: DashRequest) -> ServiceEnvelope:
    data = build_dashboard(
        req.symbol,
        use_live_alpaca=req.use_live_alpaca,
        full=req.full,
    )
    # Never expose HTML desk payload
    data.pop("html", None)
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/dashboard/{symbol}", response_model=ServiceEnvelope)
def dashboard_get(
    symbol: str,
    use_live_alpaca: bool = True,
    full: bool = False,
) -> ServiceEnvelope:
    return dashboard(
        DashRequest(symbol=symbol, use_live_alpaca=use_live_alpaca, full=full)
    )


app.include_router(router)
