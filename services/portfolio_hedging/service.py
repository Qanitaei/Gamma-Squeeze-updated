from __future__ import annotations

from typing import Any

from gamma_squeeze.config import alpaca_configured, resolve_matrix_root
from gamma_squeeze.features.feature_store import build_features_for_symbol, load_features
from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix
from gamma_squeeze.models.ensemble import SqueezeEnsemble
from gamma_squeeze.risk.portfolio_hedging_engine import (
    HEDGE_STRUCTURES,
    OPTIMIZE_OBJECTIVES,
    engine_meta,
    run_portfolio_hedging_engine,
)
from gamma_squeeze.risk.trade_recommendations import recommend_trades


def _load_matrix(symbol: str, as_of: str | None) -> tuple[dict[str, Any] | None, str]:
    """SSD → Alpaca API → KV. Never call Schwab."""
    root = resolve_matrix_root()
    dates = list_local_dates(root, symbol)
    date_used = as_of or (dates[-1] if dates else None)
    if date_used:
        local = load_local_matrix(root, symbol, date_used)
        if local:
            return local, "ssd"
    if alpaca_configured():
        try:
            from gamma_squeeze.ingest.alpaca_api import fetch_live_options_matrix

            live = fetch_live_options_matrix(symbol, as_of=as_of)
            if live.get("contracts"):
                return live, "alpaca_api"
        except Exception:  # noqa: BLE001
            pass
    try:
        from gamma_squeeze.data_sources.registry import get_adapter, load_all_adapters

        load_all_adapters()
        out = get_adapter("options.alpaca").fetch_matrix(symbol, as_of=as_of)
        matrix = (out.get("data") or {}).get("matrix")
        if matrix:
            return matrix, str((out.get("meta") or {}).get("origin", "alpaca"))
    except Exception:  # noqa: BLE001
        pass
    return None, "none"


def _composite_forecast(symbol: str, feat) -> dict[str, float] | None:
    try:
        from gamma_squeeze.models.composite_squeeze import run_composite_squeeze

        result = run_composite_squeeze(symbol, feat, matrix=None)
        return {
            "probability": float(result.probability),
            "magnitude": float(result.magnitude),
            "expected_duration": float(result.expected_duration),
            "expected_start": float(result.expected_start),
            "confidence": float(result.confidence),
        }
    except Exception:  # noqa: BLE001
        return None


def run_portfolio_hedge(
    symbol: str,
    *,
    optimize_for: str = "Risk",
    use_live_alpaca: bool = True,
    portfolio: dict[str, float] | None = None,
    use_composite_forecast: bool = True,
) -> dict[str, Any]:
    sym = symbol.upper()
    feat = load_features(sym)
    if feat.empty:
        feat = build_features_for_symbol(
            sym,
            include_skew=False,
            include_macro=True,
            include_technicals=True,
            include_dealer=True,
            include_options_metrics=True,
        )
    if feat.empty:
        return {
            "symbol": sym,
            "error": "no_features",
            "recommendations": [],
            "portfolio_hedges": [],
            "trades": [],
            "optimized": {},
            "recommend": list(HEDGE_STRUCTURES),
            "optimize": list(OPTIMIZE_OBJECTIVES),
        }

    row = feat.sort_values("as_of").iloc[-1]
    as_of = str(row["as_of"])[:10]
    matrix = None
    origin = "features"
    if use_live_alpaca:
        matrix, origin = _load_matrix(sym, as_of)

    forecast_obj = SqueezeEnsemble.load_latest().predict(
        row, symbol=sym, as_of=as_of, matrix=matrix
    )
    composite = _composite_forecast(sym, feat) if use_composite_forecast else None
    plan = run_portfolio_hedging_engine(
        sym,
        row,
        as_of=as_of,
        horizons=forecast_obj.horizons,
        forecast=composite,
        optimize_for=optimize_for,
        portfolio=portfolio,
        data_origin=origin if matrix else "features",
    )
    trades = recommend_trades(forecast_obj.horizons, row.to_dict())
    from dataclasses import asdict

    payload = plan.to_dict()
    payload.update(
        {
            "portfolio_hedges": [asdict(h) for h in plan.as_schema_hedges()],
            "trades": [asdict(t) for t in trades],
            "alpaca_configured": alpaca_configured(),
            "forecast_source": "composite" if composite else "ensemble",
            "meta": {**plan.meta, "engine": engine_meta()},
        }
    )
    return payload
