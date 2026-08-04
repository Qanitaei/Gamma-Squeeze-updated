from __future__ import annotations

from typing import Any

from gamma_squeeze.explain.engine import EXPLAIN_COMPONENTS, engine_meta, explain_prediction
from gamma_squeeze.features.feature_store import build_features_for_symbol, load_features
from gamma_squeeze.ingest.alpaca_api import alpaca_api_health
from gamma_squeeze.models.ensemble import SqueezeEnsemble
from gamma_squeeze.models.squeeze_probability import load_probability_bundle


def _composite_prediction(symbol: str, feat) -> dict[str, Any] | None:
    try:
        from gamma_squeeze.models.composite_squeeze import run_composite_squeeze

        result = run_composite_squeeze(symbol, feat, matrix=None)
        return {
            "probability": float(result.probability),
            "magnitude": float(result.magnitude),
            "expected_duration": float(result.expected_duration),
            "expected_start": float(result.expected_start),
            "confidence": float(result.confidence),
            "risk_rating": result.risk_rating,
            "horizon_days": 5,
            "scores": result.scores.to_dict(),
        }
    except Exception:  # noqa: BLE001
        return None


def run_explain(
    symbol: str,
    *,
    as_of: str | None = None,
    include_forecast: bool = True,
    use_composite: bool = True,
    use_kernel_shap: bool = False,
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
            "why": "No features available to explain.",
            "components": list(EXPLAIN_COMPONENTS),
        }

    if as_of and "as_of" in feat.columns:
        sub = feat[feat["as_of"].astype(str).str[:10] == as_of[:10]]
        row = sub.iloc[-1] if not sub.empty else feat.sort_values("as_of").iloc[-1]
    else:
        row = feat.sort_values("as_of").iloc[-1]
    as_of_s = str(row["as_of"])[:10]

    horizons = None
    prediction = None
    forecast_source = "none"
    if include_forecast:
        if use_composite:
            prediction = _composite_prediction(sym, feat)
            if prediction:
                forecast_source = "composite"
        if prediction is None:
            forecast = SqueezeEnsemble.load_latest().predict(
                row, symbol=sym, as_of=as_of_s, matrix=None
            )
            horizons = forecast.horizons
            meta = forecast.meta or {}
            # Prefer freshly built WHY over a stale embedded block
            h5 = next((h for h in horizons if h.horizon_days == 5), horizons[0])
            prediction = {
                "probability": float(h5.squeeze_probability),
                "magnitude": float(h5.expected_magnitude_pct),
                "expected_duration": float(h5.expected_duration_days),
                "confidence": float(h5.confidence_score),
                "horizon_days": int(h5.horizon_days),
            }
            forecast_source = "ensemble"
            # Keep schema explanations from ensemble as supplement
            if meta.get("explainability") and not use_composite:
                # still rebuild so every response has a fresh why
                pass

    attention = None
    try:
        from gamma_squeeze.deep.temporal_model import load_tft_model

        tft = load_tft_model()
        if tft is not None:
            fc = tft.forecast(feat)
            attention = fc.get("attention_weights")
    except Exception:  # noqa: BLE001
        attention = None

    bundle = load_probability_bundle()
    expl = explain_prediction(
        sym,
        row,
        features=feat,
        horizons=horizons,
        prediction=prediction,
        probability_bundle=bundle,
        attention=attention,
        as_of=as_of_s,
        use_kernel_shap=use_kernel_shap,
    )
    out = expl.to_dict()
    if not out.get("why"):
        out["why"] = (
            f"Prediction P={float((prediction or {}).get('probability') or 0):.3f} "
            "produced from available features; see decision_trace for stages."
        )
    out["from_forecast"] = include_forecast
    out["forecast_source"] = forecast_source
    out["alpaca"] = alpaca_api_health()
    out["rule"] = "Every prediction must explain WHY it was produced"
    return out


def explain_meta() -> dict[str, Any]:
    """Service catalog + live Alpaca API health."""
    return {
        **engine_meta(),
        "data_plane": "alpaca",
        "alpaca": alpaca_api_health(),
        "endpoints": ["/v1/meta", "/v1/explain", "/v1/explain/{symbol}"],
    }
