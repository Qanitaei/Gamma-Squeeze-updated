from __future__ import annotations

from typing import Any

from gamma_squeeze.deep.temporal_model import (
    TFT_HORIZON_LABELS,
    TFT_HORIZONS,
    TFT_OUTPUTS,
    TFT_QUANTILES,
    TFT_TARGET_LABELS,
    TFT_TARGETS,
    TFTForecastModel,
    load_tft_model,
    save_tft_model,
)
from gamma_squeeze.features.feature_store import (
    build_features_for_symbol,
    load_features,
    model_feature_matrix,
)
from gamma_squeeze.labels.label_builder import build_labels_for_symbol, load_labels


def run_tft(
    symbol: str,
    *,
    train: bool = True,
    lookback: int = 32,
    horizons: list[int] | None = None,
    targets: list[str] | None = None,
    quantile_mode: str = "median_spread",
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
        return {"symbol": sym, "error": "no_features", "horizons": {}, "attention_weights": {}}

    labs = load_labels(sym)
    if labs.empty:
        labs = build_labels_for_symbol(sym, features=feat)

    _, cols = model_feature_matrix(feat)
    use_horizons = horizons or list(TFT_HORIZONS)
    use_targets = targets or list(TFT_TARGETS)

    model = load_tft_model()
    metrics: dict[str, Any] = {"status": "infer_only"}
    if train or model is None or not getattr(model, "models", None):
        model = TFTForecastModel(
            lookback=lookback,
            horizons=list(use_horizons),
            targets=list(use_targets),
            feature_columns=cols,
            quantile_mode=quantile_mode,
        )
        metrics = model.fit(feat, labs)
        if metrics.get("status") == "ok":
            save_tft_model(model)

    assert model is not None
    out = model.forecast(feat)
    # also attach legacy paths view
    paths = model.forecast_paths(feat)
    as_of = str(feat.sort_values("as_of").iloc[-1]["as_of"])
    return {
        "symbol": sym,
        "as_of": as_of,
        "horizons": out.get("horizons", {}),
        "attention_weights": out.get("attention_weights", {}),
        "prediction_quantiles": ["q10", "q25", "median/q50", "q75", "q90"],
        "targets": out.get("targets") or use_targets,
        "target_labels": out.get("target_labels") or dict(TFT_TARGET_LABELS),
        "horizon_list": out.get("horizon_list") or use_horizons,
        "horizon_labels": out.get("horizon_labels")
        or {str(h): TFT_HORIZON_LABELS.get(int(h), f"{h}d") for h in use_horizons},
        "quantiles": out.get("quantiles") or list(TFT_QUANTILES),
        "outputs": out.get("outputs") or list(TFT_OUTPUTS),
        "version": out.get("version"),
        "backend": out.get("backend"),
        "lookback": out.get("lookback", lookback),
        "train_metrics": metrics,
        "paths": paths.get("paths"),
    }
