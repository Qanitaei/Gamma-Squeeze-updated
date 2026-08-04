from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gamma_squeeze.config import alpaca_configured, alpaca_credentials, resolve_matrix_root
from gamma_squeeze.features.feature_store import build_features_for_symbol, load_features
from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix
from gamma_squeeze.models.composite_squeeze import (
    COMPOSITE_INPUT_LABELS,
    COMPOSITE_INPUTS,
    COMPOSITE_OUTPUT_LABELS,
    COMPOSITE_OUTPUTS,
    COMPOSITE_SCORE_LABELS,
    COMPOSITE_SCORES,
    run_composite_squeeze,
)
from gamma_squeeze.models.ensemble import SqueezeEnsemble
from gamma_squeeze.serve.export_forecasts import export_forecast, forecast_paths
from gamma_squeeze.serve.schemas import validate_forecast_dict


def _load_options_matrix(symbol: str, as_of: str | None = None) -> tuple[dict[str, Any] | None, str | None, str]:
    """Load options matrix: local SSD → Alpaca API → alpaca-options-matrix-backup KV."""
    sym = symbol.upper()
    root = resolve_matrix_root()
    dates = list_local_dates(root, sym)
    date_used = as_of or (dates[-1] if dates else None)
    if date_used:
        matrix = load_local_matrix(root, sym, date_used)
        if matrix:
            return matrix, date_used, "ssd"
    try:
        from gamma_squeeze.data_sources.registry import get_adapter, load_all_adapters

        load_all_adapters()
        out = get_adapter("options.alpaca").fetch_matrix(sym, as_of=as_of)
        matrix = (out.get("data") or {}).get("matrix")
        if matrix:
            origin = (out.get("meta") or {}).get("origin", "alpaca")
            used = out.get("as_of") or as_of or matrix.get("as_of_date")
            return matrix, used, str(origin)
    except Exception:  # noqa: BLE001
        pass
    return None, date_used, "none"


def _export_forecast_dict(payload: dict[str, Any]) -> Path:
    """Persist a validated forecast dict (composite-enriched) to SSD/local."""
    dated, latest = forecast_paths(str(payload["symbol"]), str(payload["as_of"]))
    for path in (dated, latest):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
    return dated


def run_squeeze(
    symbol: str,
    *,
    persist: bool = True,
    composite: bool = True,
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
            "alpaca_configured": alpaca_configured(),
            "inputs": list(COMPOSITE_INPUTS),
            "scores": list(COMPOSITE_SCORES),
            "outputs": list(COMPOSITE_OUTPUTS),
        }

    row = feat.sort_values("as_of").iloc[-1]
    as_of = str(row["as_of"])[:10]
    matrix, matrix_as_of, origin = _load_options_matrix(sym, as_of=as_of)
    creds = alpaca_credentials()

    if composite:
        result = run_composite_squeeze(sym, feat, matrix)
        payload = result.forecast
        errors = validate_forecast_dict(payload) if payload else ["empty_forecast"]
        path = None
        if persist and payload and not errors:
            path = str(_export_forecast_dict(payload))
        out = result.to_dict()
        out.update(
            {
                "validation_errors": errors,
                "export_path": path,
                "alpaca_configured": alpaca_configured(),
                "data_sources": {
                    "alpaca_trading_api": creds.get("trading_base"),
                    "alpaca_paper": creds.get("paper"),
                    "matrix_origin": origin if matrix else None,
                    "matrix_as_of": matrix_as_of,
                },
                "contract": {
                    "inputs": list(COMPOSITE_INPUTS),
                    "input_labels": dict(COMPOSITE_INPUT_LABELS),
                    "scores": list(COMPOSITE_SCORES),
                    "score_labels": dict(COMPOSITE_SCORE_LABELS),
                    "outputs": list(COMPOSITE_OUTPUTS),
                    "output_labels": dict(COMPOSITE_OUTPUT_LABELS),
                },
            }
        )
        return out

    ensemble = SqueezeEnsemble.load_latest()
    forecast = ensemble.predict(row, symbol=sym, as_of=as_of, matrix=matrix)
    payload = forecast.to_dict()
    errors = validate_forecast_dict(payload)
    path = None
    if persist and not errors:
        path = str(export_forecast(forecast))
    return {
        "symbol": sym,
        "as_of": as_of,
        "forecast": payload,
        "validation_errors": errors,
        "export_path": path,
        "alpaca_configured": alpaca_configured(),
    }
