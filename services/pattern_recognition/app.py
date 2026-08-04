"""Technical Pattern Recognition API — port 8003."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fastapi import APIRouter
from pydantic import BaseModel, Field

from gamma_squeeze.patterns.recognition import (
    CANDLESTICK_PATTERNS,
    CHART_PATTERNS,
    PATTERN_BACKEND,
    PATTERN_OUTPUTS,
)
from services.common.app_factory import create_service_app
from services.common.schemas import ServiceEnvelope
from services.pattern_recognition.service import run_patterns

SERVICE = "pattern_recognition"
app = create_service_app(
    name=SERVICE,
    description=(
        "CNN/LSTM/Transformer hybrid technical pattern recognition — "
        "chart patterns, candlestick library; returns pattern, confidence, "
        "target projection, and probability"
    ),
)
router = APIRouter(prefix="/v1", tags=[SERVICE])


class PatternRequest(BaseModel):
    symbol: str
    lookback: int = Field(80, ge=20, le=252)
    min_confidence: float = Field(0.40, ge=0.0, le=1.0)
    include_candles: bool = True


@router.get("/catalog")
def catalog() -> dict:
    return {
        "chart_patterns": list(CHART_PATTERNS),
        "candlestick_library": list(CANDLESTICK_PATTERNS),
        "outputs": list(PATTERN_OUTPUTS),
        "backend": PATTERN_BACKEND,
    }


@router.get("/meta")
def meta() -> dict:
    return {
        "service": SERVICE,
        "backend": PATTERN_BACKEND,
        "hybrid_branches": ["cnn", "lstm", "transformer"],
        "chart_patterns": list(CHART_PATTERNS),
        "candlestick_library": list(CANDLESTICK_PATTERNS),
        "outputs": list(PATTERN_OUTPUTS),
        "return_fields": {
            "pattern": "Detected pattern name",
            "confidence": "Fused rule + hybrid score [0,1]",
            "target_projection": "price / pct / direction",
            "probability": "Continuation / completion probability [0,1]",
        },
    }


@router.post("/patterns", response_model=ServiceEnvelope)
def patterns(req: PatternRequest) -> ServiceEnvelope:
    data = run_patterns(
        req.symbol,
        lookback=req.lookback,
        min_confidence=req.min_confidence,
        include_candles=req.include_candles,
    )
    return ServiceEnvelope(
        service=SERVICE,
        symbol=req.symbol.upper(),
        as_of=data.get("as_of"),
        degraded=bool(data.get("error")),
        upstream_errors=[data["error"]] if data.get("error") else [],
        data=data,
    )


@router.get("/patterns/{symbol}", response_model=ServiceEnvelope)
def patterns_get(
    symbol: str,
    lookback: int = 80,
    min_confidence: float = 0.40,
    include_candles: bool = True,
) -> ServiceEnvelope:
    return patterns(
        PatternRequest(
            symbol=symbol,
            lookback=lookback,
            min_confidence=min_confidence,
            include_candles=include_candles,
        )
    )


app.include_router(router)
