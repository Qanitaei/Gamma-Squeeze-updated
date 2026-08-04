from __future__ import annotations

from typing import Any

from gamma_squeeze.features.feature_store import (
    build_features_for_symbol,
    load_features,
    model_feature_matrix,
)
from gamma_squeeze.labels.label_builder import build_labels_for_symbol, load_labels
from gamma_squeeze.models.xgboost_direction import (
    DIRECTION_HORIZON_LABELS,
    DIRECTION_HORIZONS,
    DIRECTION_TARGETS,
    DIRECTION_TRAINING,
    load_direction_model,
    save_direction_model,
    train_direction_model,
)


def run_direction(
    symbol: str,
    *,
    train: bool = True,
    horizon: int | None = None,
    horizons: list[int] | None = None,
    n_trials: int = 15,
    cv_folds: int = 5,
    include_shap: bool = True,
    direction_threshold: float = 0.005,
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
    labs = load_labels(sym)
    if labs.empty and not feat.empty:
        labs = build_labels_for_symbol(sym, features=feat)
    if feat.empty:
        return {"symbol": sym, "error": "no_features"}

    _, cols = model_feature_matrix(feat)
    use_horizons = horizons or list(DIRECTION_HORIZONS)

    bundle = load_direction_model()
    trained = False
    if train or bundle is None or not getattr(bundle, "models", None):
        bundle = train_direction_model(
            feat,
            labs,
            cols,
            horizons=use_horizons,
            horizon=horizon,
            n_trials=n_trials,
            cv_folds=cv_folds,
            include_shap=include_shap,
            direction_threshold=direction_threshold,
        )
        if bundle.models:
            save_direction_model(bundle)
            trained = True

    row = feat.sort_values("as_of").iloc[-1]
    if horizon is not None:
        pred = bundle.predict_row(row, horizon=int(horizon))
        payload = {
            "symbol": sym,
            "as_of": str(row["as_of"]),
            "horizon": int(horizon),
            "classification": pred.get("classification"),
            "regression": pred.get("regression"),
            "probability": pred.get("probability"),
            "confidence": pred.get("confidence"),
            "direction": pred.get("direction"),
            "trained": trained or train,
            "version": getattr(bundle, "version", None),
            "cv_scores": _horizon_meta(bundle, int(horizon), "cv_scores"),
            "best_params": _horizon_meta(bundle, int(horizon), "best_params"),
            "shap": _horizon_meta(bundle, int(horizon), "shap"),
        }
        return payload

    pred = bundle.predict_row(row)
    cv_all = {f"{h}d": hb.cv_scores for h, hb in bundle.models.items()}
    params_all = {f"{h}d": hb.best_params for h, hb in bundle.models.items()}
    shap_all = {
        f"{h}d": {
            "top_features": (hb.shap.get("classification") or {}).get("top_features", [])[:10],
            "status": (hb.shap.get("classification") or {}).get("status"),
        }
        for h, hb in bundle.models.items()
    }
    return {
        "symbol": sym,
        "as_of": str(row["as_of"]),
        "horizons": pred.get("horizons", {}),
        "classification": pred.get("classification"),
        "regression": pred.get("regression"),
        "probability": pred.get("probability"),
        "direction": pred.get("direction"),
        "proba": pred.get("proba"),
        "trained": trained or train,
        "version": getattr(bundle, "version", None),
        "horizon_list": list(getattr(bundle, "horizons", use_horizons)),
        "horizon_labels": {str(h): DIRECTION_HORIZON_LABELS.get(int(h), f"{h}d") for h in use_horizons},
        "targets": list(DIRECTION_TARGETS),
        "training": list(DIRECTION_TRAINING),
        "cv_scores": cv_all,
        "best_params": params_all,
        "shap": shap_all,
        "backend": getattr(bundle, "backend", None),
    }


def _horizon_meta(bundle: Any, horizon: int, field: str) -> Any:
    models = getattr(bundle, "models", {}) or {}
    hb = models.get(int(horizon))
    if hb is None:
        return None
    return getattr(hb, field, None)
