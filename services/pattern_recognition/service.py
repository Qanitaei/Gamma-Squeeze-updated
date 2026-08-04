from __future__ import annotations

from typing import Any

from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv
from gamma_squeeze.patterns.recognition import (
    CANDLESTICK_PATTERNS,
    CHART_PATTERNS,
    PATTERN_BACKEND,
    PATTERN_OUTPUTS,
    detect_patterns,
)


def run_patterns(
    symbol: str,
    *,
    lookback: int = 80,
    min_confidence: float = 0.40,
    include_candles: bool = True,
) -> dict[str, Any]:
    sym = symbol.upper()
    try:
        ohlcv = fetch_daily_ohlcv(sym)
        result = detect_patterns(
            ohlcv,
            lookback=lookback,
            min_confidence=min_confidence,
            include_candles=include_candles,
        )
        result["symbol"] = sym
        result["as_of"] = str(ohlcv.index[-1]) if len(ohlcv) else None
        result.setdefault("outputs", list(PATTERN_OUTPUTS))
        result.setdefault("backend", PATTERN_BACKEND)
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "symbol": sym,
            "patterns": [],
            "summary": {},
            "error": str(exc),
            "backend": PATTERN_BACKEND,
            "outputs": list(PATTERN_OUTPUTS),
            "catalog": {
                "chart": list(CHART_PATTERNS),
                "candlestick": list(CANDLESTICK_PATTERNS),
            },
        }
