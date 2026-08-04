"""SHAP explainability adapter (falls back to heuristic attributions)."""

from __future__ import annotations

from typing import Any

import numpy as np


def shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    *,
    max_samples: int = 100,
) -> dict[str, Any]:
    try:
        import shap
    except ImportError:
        return _fallback_importances(model, X, feature_names, status="shap_not_installed")

    if len(X) == 0:
        return {"status": "empty", "importances": {}, "top_features": []}

    sample = np.asarray(X[:max_samples], dtype=float)
    try:
        values = None
        # Tree models (XGBoost / LGBM / sklearn HGB)
        try:
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(sample)
            if isinstance(raw, list):
                # multiclass → mean |shap| across classes
                values = np.mean([np.abs(v) for v in raw], axis=0)
            else:
                values = np.abs(raw)
        except Exception:  # noqa: BLE001
            explainer = shap.Explainer(model.predict, sample)
            explanation = explainer(sample)
            values = np.abs(explanation.values)
            if values.ndim == 3:
                values = values.mean(axis=-1)

        mean_abs = np.asarray(values, dtype=float).mean(axis=0).reshape(-1)
        importances = {
            feature_names[i]: float(mean_abs[i])
            for i in range(min(len(feature_names), len(mean_abs)))
        }
        top = sorted(importances.items(), key=lambda kv: -abs(kv[1]))[:15]
        return {
            "status": "ok",
            "importances": importances,
            "top_features": [{"feature": k, "mean_abs_shap": v} for k, v in top],
        }
    except Exception as exc:  # noqa: BLE001
        out = _fallback_importances(model, sample, feature_names, status="error")
        out["error"] = str(exc)
        return out


def _fallback_importances(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    *,
    status: str,
) -> dict[str, Any]:
    importances: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        fi = np.asarray(model.feature_importances_, dtype=float)
        importances = {
            feature_names[i]: float(fi[i])
            for i in range(min(len(feature_names), len(fi)))
        }
    top = sorted(importances.items(), key=lambda kv: -abs(kv[1]))[:15]
    return {
        "status": status,
        "importances": importances,
        "top_features": [{"feature": k, "mean_abs_shap": v} for k, v in top],
    }
