"""Alert Engine API — port 8012 (REST + WebSockets + multi-channel notify)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from gamma_squeeze.stack.kafka_bus import publish_json
from gamma_squeeze.stack.redis_client import publish as redis_publish
from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.trade_alerts.service import alerts_meta, evaluate_alerts

SERVICE = "trade_alerts"
app = create_service_app(
    name=SERVICE,
    description="Alert Engine: squeeze/dealer/walls/IV/patterns/earnings → WS/Email/SMS/Discord/Slack/Webhook",
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class AlertRequest(BaseModel):
    symbol: str
    min_probability: float = Field(0.55, ge=0.0, le=1.0)
    min_confidence: float = Field(0.45, ge=0.0, le=1.0)
    horizon_days: int = Field(5, ge=1, le=10)
    notify: bool = False
    channels: list[str] | None = None
    use_live_alpaca: bool = True
    dealer_flip_distance_pct: float | None = None
    gamma_ramp_abs: float | None = None
    wall_break_buffer_pct: float | None = None
    iv_expansion_ratio: float | None = None
    dealer_hedge_shares: float | None = None
    pattern_min_confidence: float | None = None


def _emit(data: dict) -> None:
    payload = json.dumps(data)
    redis_publish("gamma.alerts", payload)
    publish_json("alerts", data, key=data.get("symbol"))


@router.get("/meta")
def meta() -> dict:
    """Alpaca API health + trigger/channel catalog."""
    return alerts_meta()


@router.post("/alerts", response_model=ServiceEnvelope)
def alerts(req: AlertRequest) -> ServiceEnvelope:
    overrides = {
        k: v
        for k, v in {
            "dealer_flip_distance_pct": req.dealer_flip_distance_pct,
            "gamma_ramp_abs": req.gamma_ramp_abs,
            "wall_break_buffer_pct": req.wall_break_buffer_pct,
            "iv_expansion_ratio": req.iv_expansion_ratio,
            "dealer_hedge_shares": req.dealer_hedge_shares,
            "pattern_min_confidence": req.pattern_min_confidence,
        }.items()
        if v is not None
    }
    data = evaluate_alerts(
        req.symbol,
        min_probability=req.min_probability,
        min_confidence=req.min_confidence,
        horizon_days=req.horizon_days,
        notify=req.notify,
        channels=req.channels,
        use_live_alpaca=req.use_live_alpaca,
        **overrides,
    )
    if data.get("fired"):
        _emit(data)
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/alerts/{symbol}", response_model=ServiceEnvelope)
def alerts_get(
    symbol: str,
    min_probability: float = 0.55,
    min_confidence: float = 0.45,
    horizon_days: int = 5,
    notify: bool = False,
    use_live_alpaca: bool = True,
) -> ServiceEnvelope:
    return alerts(
        AlertRequest(
            symbol=symbol,
            min_probability=min_probability,
            min_confidence=min_confidence,
            horizon_days=horizon_days,
            notify=notify,
            use_live_alpaca=use_live_alpaca,
        )
    )


@router.post("/notify", response_model=ServiceEnvelope)
def notify(req: AlertRequest) -> ServiceEnvelope:
    req.notify = True
    return alerts(req)


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket, symbol: str = "AAPL", interval_s: float = 15.0):
    """Stream Alert Engine evaluations over WebSockets."""
    await websocket.accept()
    try:
        while True:
            data = evaluate_alerts(symbol, notify=False)
            await websocket.send_json(data)
            await asyncio.sleep(max(1.0, interval_s))
    except WebSocketDisconnect:
        return


app.include_router(router)
