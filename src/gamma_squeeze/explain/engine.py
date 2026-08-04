"""Explainability Engine — every prediction must explain WHY.

Provides: SHAP, Attention Maps, Feature Importance, Counterfactuals,
Partial Dependence, Decision Trace, Model Confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable  # noqa: F401 — Any used throughout

import numpy as np
import pandas as pd

from gamma_squeeze.explain.attributions import FEATURE_WEIGHTS, explain_row
from gamma_squeeze.serve.schemas import Explanation, HorizonForecast

try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    shap = None  # type: ignore
    HAS_SHAP = False


VERSION = "explainability-v1"

EXPLAIN_COMPONENTS = (
    "SHAP",
    "Attention Maps",
    "Feature Importance",
    "Counterfactual Analysis",
    "Partial Dependence",
    "Decision Trace",
    "Model Confidence",
)


@dataclass
class ExplanationBundle:
    """Full why-package attached to a prediction."""

    symbol: str
    as_of: str
    prediction: dict[str, Any]
    why: str
    shap: dict[str, Any] = field(default_factory=dict)
    attention_maps: dict[str, Any] = field(default_factory=dict)
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    partial_dependence: dict[str, Any] = field(default_factory=dict)
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    model_confidence: dict[str, Any] = field(default_factory=dict)
    schema_explanations: list[Explanation] = field(default_factory=list)
    version: str = VERSION
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of,
            "prediction": self.prediction,
            "why": self.why,
            "shap": self.shap,
            "attention_maps": self.attention_maps,
            "feature_importance": self.feature_importance,
            "counterfactuals": self.counterfactuals,
            "partial_dependence": self.partial_dependence,
            "decision_trace": self.decision_trace,
            "model_confidence": self.model_confidence,
            "explanations": [asdict(e) for e in self.schema_explanations],
            "version": self.version,
            "meta": self.meta,
            "components": list(EXPLAIN_COMPONENTS),
        }


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _numeric_frame(features: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    skip = {"symbol", "as_of", "regime"}
    if columns:
        cols = [c for c in columns if c in features.columns]
    else:
        cols = [
            c
            for c in features.columns
            if c not in skip and pd.api.types.is_numeric_dtype(features[c])
        ]
    if not cols:
        # force coerce preferred
        preferred = [c for c in FEATURE_WEIGHTS if c in features.columns]
        cols = preferred
    X = features[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if cols else pd.DataFrame()
    return X, cols


def _predict_fn_from_bundle(bundle) -> Callable[[np.ndarray], np.ndarray] | None:
    if bundle is None or not getattr(bundle, "models", None):
        return None
    # Prefer 5d head
    model = bundle.models.get(5) or next(iter(bundle.models.values()), None)
    if model is None:
        return None

    def _fn(X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # class 1 = squeeze / positive
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.reshape(-1)
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X), dtype=float).reshape(-1)
        return np.zeros(len(X), dtype=float)

    return _fn


def _heuristic_predict(row: dict[str, Any]) -> float:
    stress = _f(row, "positioning_stress")
    neg = _f(row, "regime_neg_gamma")
    gex = _f(row, "net_gex", _f(row, "gex_per_spot"))
    gex_term = float(np.clip(-np.tanh(gex / 1e6), 0, 1))
    return float(np.clip(0.45 * stress + 0.35 * neg + 0.20 * gex_term, 0, 1))


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _pack_shap_values(
    columns: list[str],
    x: np.ndarray,
    vals: np.ndarray,
    *,
    base: float,
    backend: str,
    top_k: int,
) -> dict[str, Any]:
    pairs = sorted(
        [
            {
                "feature": columns[i],
                "shap_value": float(vals[i]),
                "value": float(x[0, i]),
                "direction": "supports" if vals[i] > 0 else "opposes",
            }
            for i in range(min(len(columns), len(vals)))
        ],
        key=lambda d: -abs(d["shap_value"]),
    )
    return {
        "backend": backend,
        "base_value": float(base),
        "values": pairs[:top_k],
        "sum_shap": float(np.asarray(vals, dtype=float).sum()),
    }


def compute_shap(
    row: dict[str, Any],
    background: pd.DataFrame,
    columns: list[str],
    predict_fn: Callable[[np.ndarray], np.ndarray] | None,
    *,
    model: Any = None,
    top_k: int = 12,
    use_kernel_shap: bool = False,
) -> dict[str, Any]:
    x = np.asarray([[_f(row, c) for c in columns]], dtype=float) if columns else np.zeros((1, 0))
    if not columns:
        return {"backend": "none", "values": [], "base_value": 0.0}

    shap_err: str | None = None

    # 1) Fast TreeExplainer when model is tree-based
    if HAS_SHAP and model is not None:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            vals = np.asarray(sv, dtype=float).reshape(-1)
            base_raw = explainer.expected_value
            if isinstance(base_raw, (list, np.ndarray)):
                base = float(np.asarray(base_raw).reshape(-1)[-1])
            else:
                base = float(base_raw)
            return _pack_shap_values(columns, x, vals, base=base, backend="shap.TreeExplainer", top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            shap_err = f"tree:{exc}"

    # 2) Optional KernelExplainer (slow — opt-in only)
    if HAS_SHAP and use_kernel_shap and predict_fn is not None and len(background) >= 5:
        try:
            bg = background[columns].to_numpy(dtype=float)
            bg = bg[-min(40, len(bg)) :]
            explainer = shap.KernelExplainer(predict_fn, bg)
            sv = explainer.shap_values(x, nsamples=min(64, max(16, len(columns) * 2)))
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            vals = np.asarray(sv, dtype=float).reshape(-1)
            base = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
            return _pack_shap_values(columns, x, vals, base=base, backend="shap.KernelExplainer", top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            shap_err = f"kernel:{exc}"

    # 3) Linear / weight fallback (always available, local SHAP-style)
    pairs = []
    for feat, weight in FEATURE_WEIGHTS.items():
        if feat not in columns and feat not in row:
            continue
        val = _f(row, feat)
        if feat == "gex_per_spot":
            norm = max(min(val / 1e6, 3.0), -3.0)
        elif feat == "pcr_oi":
            norm = val - 1.0
        else:
            norm = val
        contrib = float(weight * norm)
        pairs.append(
            {
                "feature": feat,
                "shap_value": contrib,
                "value": val,
                "direction": "supports" if contrib > 0 else "opposes",
            }
        )
    pairs.sort(key=lambda d: -abs(d["shap_value"]))
    return {
        "backend": "linear_attribution",
        "base_value": 0.35,
        "values": pairs[:top_k],
        "sum_shap": float(sum(p["shap_value"] for p in pairs)),
        "warning": shap_err,
    }


# ---------------------------------------------------------------------------
# Attention maps
# ---------------------------------------------------------------------------

def compute_attention_maps(
    features: pd.DataFrame,
    columns: list[str],
    *,
    lookback: int = 16,
    external_attention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if external_attention:
        return {"backend": "model", **external_attention}

    if features.empty or not columns:
        return {"backend": "empty", "temporal": [], "variables": []}

    tail = features.tail(lookback)
    X, cols = _numeric_frame(tail, columns)
    if X.empty:
        return {"backend": "empty", "temporal": [], "variables": []}

    arr = X.to_numpy(dtype=float)
    # Temporal: emphasize recent bars via softmax of recency + |return proxy|
    t = arr.shape[0]
    recency = np.linspace(0.2, 1.0, t)
    if "ret_1d" in X.columns:
        shock = np.abs(X["ret_1d"].to_numpy())
        logits = recency + 0.5 * (shock / (shock.max() + 1e-9))
    else:
        logits = recency
    t_w = np.exp(logits - logits.max())
    t_w = t_w / t_w.sum()
    dates = [str(x)[:10] for x in tail.get("as_of", tail.index).tolist()]

    # Variable attention: |z| of latest vs history std
    latest = arr[-1]
    std = arr.std(axis=0) + 1e-9
    mean = arr.mean(axis=0)
    z = np.abs((latest - mean) / std)
    v_w = z / (z.sum() + 1e-9)
    variables = sorted(
        [{"feature": cols[i], "weight": float(v_w[i])} for i in range(len(cols))],
        key=lambda d: -d["weight"],
    )
    return {
        "backend": "recency_zscore",
        "temporal": [
            {"lag": int(t - 1 - i), "as_of": dates[i] if i < len(dates) else None, "weight": float(t_w[i])}
            for i in range(t)
        ],
        "variables": variables[:15],
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    bundle,
    columns: list[str],
    shap_block: dict[str, Any],
    *,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    # Model native importances
    model = None
    if bundle is not None and getattr(bundle, "models", None):
        model = bundle.models.get(5) or next(iter(bundle.models.values()), None)
    if model is not None and hasattr(model, "feature_importances_") and columns:
        imp = np.asarray(model.feature_importances_, dtype=float)
        n = min(len(imp), len(columns))
        pairs = [
            {"feature": columns[i], "importance": float(imp[i]), "source": "model"}
            for i in range(n)
        ]
        pairs.sort(key=lambda d: -d["importance"])
        return pairs[:top_k]

    # From |SHAP|
    vals = shap_block.get("values") or []
    if vals:
        total = sum(abs(v["shap_value"]) for v in vals) or 1.0
        return [
            {
                "feature": v["feature"],
                "importance": float(abs(v["shap_value"]) / total),
                "source": "shap_normalized",
            }
            for v in vals[:top_k]
        ]

    # Prior weights
    total_w = sum(abs(w) for w in FEATURE_WEIGHTS.values()) or 1.0
    return [
        {"feature": f, "importance": abs(w) / total_w, "source": "prior"}
        for f, w in sorted(FEATURE_WEIGHTS.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    ]


# ---------------------------------------------------------------------------
# Counterfactuals
# ---------------------------------------------------------------------------

def compute_counterfactuals(
    row: dict[str, Any],
    columns: list[str],
    predict_fn: Callable[[np.ndarray], np.ndarray] | None,
    *,
    base_pred: float,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Find small feature flips that would change the decision most."""
    target = 0.5
    want_up = base_pred < target
    candidates = [
        "positioning_stress",
        "regime_neg_gamma",
        "net_gex",
        "gex_per_spot",
        "pcr_oi",
        "atm_iv",
        "rvol_10d",
        "ret_5d",
        "dealer_exhaustion",
        "flag_fragile_gamma",
    ]
    feats = [c for c in candidates if c in row or c in columns]
    if not feats:
        feats = columns[:8]

    def _pred(r: dict[str, Any]) -> float:
        if predict_fn is not None and columns:
            x = np.asarray([[_f(r, c) for c in columns]], dtype=float)
            return float(predict_fn(x)[0])
        return _heuristic_predict(r)

    out: list[dict[str, Any]] = []
    for feat in feats:
        trial = dict(row)
        cur = _f(row, feat)
        # Propose flip toward opposite thesis
        if feat in ("net_gex", "gex_per_spot"):
            new_val = -abs(cur) * 1.5 - 1e5 if want_up else abs(cur) * 1.5 + 1e5
        elif feat in ("positioning_stress", "regime_neg_gamma", "flag_fragile_gamma", "dealer_exhaustion"):
            new_val = 0.9 if want_up else 0.1
        elif feat == "pcr_oi":
            new_val = 0.7 if want_up else 1.4
        elif feat in ("atm_iv", "rvol_10d"):
            new_val = cur * (1.25 if want_up else 0.8)
        else:
            new_val = cur + (0.05 if want_up else -0.05)
        trial[feat] = new_val
        new_p = _pred(trial)
        delta = new_p - base_pred
        if abs(delta) < 1e-4:
            continue
        out.append(
            {
                "feature": feat,
                "from": cur,
                "to": float(new_val),
                "prediction_from": float(base_pred),
                "prediction_to": float(new_p),
                "delta": float(delta),
                "flips_decision": (base_pred - 0.5) * (new_p - 0.5) < 0,
                "statement": (
                    f"If {feat} changed {cur:.4g} → {new_val:.4g}, "
                    f"P would move {base_pred:.3f} → {new_p:.3f} (Δ={delta:+.3f})"
                ),
            }
        )
    out.sort(key=lambda d: -abs(d["delta"]))
    return out[:n]


# ---------------------------------------------------------------------------
# Partial dependence
# ---------------------------------------------------------------------------

def compute_partial_dependence(
    row: dict[str, Any],
    background: pd.DataFrame,
    columns: list[str],
    predict_fn: Callable[[np.ndarray], np.ndarray] | None,
    *,
    features: list[str] | None = None,
    grid_size: int = 9,
) -> dict[str, Any]:
    focus = features or [
        c
        for c in (
            "positioning_stress",
            "net_gex",
            "atm_iv",
            "rvol_10d",
            "regime_neg_gamma",
        )
        if c in columns or c in row
    ]
    focus = focus[:4]
    curves: dict[str, Any] = {}

    def _pred_matrix(mat: np.ndarray) -> np.ndarray:
        if predict_fn is not None:
            return predict_fn(mat)
        # heuristic per row using column alignment
        preds = []
        for i in range(mat.shape[0]):
            r = {columns[j]: float(mat[i, j]) for j in range(len(columns))}
            preds.append(_heuristic_predict(r))
        return np.asarray(preds, dtype=float)

    base_vec = np.asarray([_f(row, c) for c in columns], dtype=float) if columns else np.zeros(0)

    for feat in focus:
        if columns and feat in columns:
            idx = columns.index(feat)
            series = background[feat] if feat in background.columns else None
            if series is not None and len(series):
                lo, hi = float(np.nanpercentile(series, 5)), float(np.nanpercentile(series, 95))
            else:
                lo, hi = float(base_vec[idx] - 1), float(base_vec[idx] + 1)
            if lo == hi:
                lo, hi = lo - 1.0, hi + 1.0
            grid = np.linspace(lo, hi, grid_size)
            preds = []
            for g in grid:
                x = base_vec.copy()
                x[idx] = g
                preds.append(float(_pred_matrix(x.reshape(1, -1))[0]))
            curves[feat] = {
                "grid": [float(g) for g in grid],
                "pd": preds,
                "anchor_value": float(base_vec[idx]),
                "anchor_prediction": float(_pred_matrix(base_vec.reshape(1, -1))[0]),
            }
        else:
            # scalar heuristic sweep
            cur = _f(row, feat)
            grid = np.linspace(0.0, 1.0, grid_size) if feat.endswith("stress") or "regime" in feat else np.linspace(cur * 0.5, cur * 1.5 + 1e-6, grid_size)
            preds = []
            for g in grid:
                r = dict(row)
                r[feat] = float(g)
                preds.append(_heuristic_predict(r))
            curves[feat] = {
                "grid": [float(g) for g in grid],
                "pd": preds,
                "anchor_value": cur,
                "anchor_prediction": _heuristic_predict(row),
            }
    return {"backend": "ice_on_instance", "curves": curves}


# ---------------------------------------------------------------------------
# Decision trace
# ---------------------------------------------------------------------------

def build_decision_trace(
    row: dict[str, Any],
    prediction: dict[str, Any],
    shap_block: dict[str, Any],
    confidence: dict[str, Any],
) -> list[dict[str, Any]]:
    p = float(prediction.get("probability") or prediction.get("squeeze_probability") or 0.0)
    steps: list[dict[str, Any]] = [
        {
            "step": 1,
            "stage": "inputs",
            "detail": (
                f"Loaded features as_of={row.get('as_of')}; "
                f"stress={_f(row,'positioning_stress'):.2f}, "
                f"neg_gamma={_f(row,'regime_neg_gamma'):.2f}, "
                f"net_gex={_f(row,'net_gex'):.3g}"
            ),
        },
        {
            "step": 2,
            "stage": "model_inference",
            "detail": f"Primary prediction P={p:.3f}; magnitude={prediction.get('magnitude')}; "
            f"horizon={prediction.get('horizon_days', 5)}",
        },
    ]
    top = (shap_block.get("values") or [])[:3]
    if top:
        steps.append(
            {
                "step": 3,
                "stage": "attribution",
                "detail": "Top drivers: "
                + ", ".join(f"{t['feature']}({t['shap_value']:+.3f})" for t in top),
            }
        )
    steps.append(
        {
            "step": 4,
            "stage": "confidence",
            "detail": (
                f"Confidence={confidence.get('score', 0):.3f} "
                f"({confidence.get('rating', 'n/a')}); "
                f"components={confidence.get('components', {})}"
            ),
        }
    )
    decision = "lean_squeeze" if p >= 0.55 else ("watch" if p >= 0.4 else "no_squeeze")
    steps.append(
        {
            "step": 5,
            "stage": "decision",
            "detail": f"Decision={decision} because P={p:.3f} with confidence {confidence.get('score', 0):.2f}",
            "decision": decision,
        }
    )
    return steps


# ---------------------------------------------------------------------------
# Model confidence
# ---------------------------------------------------------------------------

def compute_model_confidence(
    prediction: dict[str, Any],
    row: dict[str, Any],
    shap_block: dict[str, Any],
    *,
    horizon_confidence: float | None = None,
) -> dict[str, Any]:
    p = float(prediction.get("probability") or prediction.get("squeeze_probability") or 0.0)
    stress = _f(row, "positioning_stress")
    agreement = 1.0 - abs(p - stress)
    # Attribution stability: concentration of |SHAP|
    vals = [abs(v["shap_value"]) for v in (shap_block.get("values") or [])]
    if vals:
        s = sum(vals) or 1.0
        share = sorted((v / s for v in vals), reverse=True)
        # top-3 concentration — moderate concentration preferred
        top3 = sum(share[:3])
        stability = float(np.clip(1.0 - abs(top3 - 0.65), 0, 1))
    else:
        stability = 0.4
    hc = float(horizon_confidence if horizon_confidence is not None else prediction.get("confidence") or 0.5)
    score = float(np.clip(0.40 * hc + 0.30 * agreement + 0.20 * stability + 0.10 * (1.0 - abs(p - 0.5) * 0.5), 0, 1))
    if score >= 0.75:
        rating = "High"
    elif score >= 0.55:
        rating = "Medium"
    elif score >= 0.35:
        rating = "Low"
    else:
        rating = "Very Low"
    return {
        "score": score,
        "rating": rating,
        "components": {
            "horizon_confidence": hc,
            "feature_model_agreement": float(agreement),
            "attribution_stability": stability,
            "probability": p,
        },
    }


def _why_narrative(
    prediction: dict[str, Any],
    shap_block: dict[str, Any],
    confidence: dict[str, Any],
    counterfactuals: list[dict[str, Any]],
) -> str:
    p = float(prediction.get("probability") or prediction.get("squeeze_probability") or 0.0)
    stance = "elevated squeeze odds" if p >= 0.55 else ("borderline odds" if p >= 0.4 else "low squeeze odds")
    drivers = shap_block.get("values") or []
    if drivers:
        top = drivers[0]
        drive = f"Largest driver: {top['feature']} ({top['shap_value']:+.3f}, {top['direction']})."
    else:
        drive = "Drivers unavailable."
    cf = ""
    if counterfactuals:
        cf = " Counterfactual: " + counterfactuals[0]["statement"]
    return (
        f"Prediction shows {stance} (P={p:.3f}) with {confidence.get('rating', 'n/a')} "
        f"confidence ({confidence.get('score', 0):.2f}). {drive}{cf}"
    )


def explain_prediction(
    symbol: str,
    feature_row: dict[str, Any] | pd.Series,
    *,
    features: pd.DataFrame | None = None,
    horizons: list[HorizonForecast] | None = None,
    prediction: dict[str, Any] | None = None,
    probability_bundle=None,
    attention: dict[str, Any] | None = None,
    as_of: str | None = None,
    use_kernel_shap: bool = False,
) -> ExplanationBundle:
    """Build a complete explanation bundle for one prediction."""
    row = feature_row.to_dict() if isinstance(feature_row, pd.Series) else dict(feature_row)
    as_of_s = (as_of or str(row.get("as_of") or ""))[:10]
    feat_df = features if features is not None else pd.DataFrame([row])

    # Resolve prediction summary
    pred = dict(prediction or {})
    if not pred and horizons:
        h5 = next((h for h in horizons if h.horizon_days == 5), horizons[0])
        pred = {
            "probability": float(h5.squeeze_probability),
            "squeeze_probability": float(h5.squeeze_probability),
            "magnitude": float(h5.expected_magnitude_pct),
            "expected_duration": float(h5.expected_duration_days),
            "confidence": float(h5.confidence_score),
            "horizon_days": int(h5.horizon_days),
        }
    if "probability" not in pred:
        pred["probability"] = _heuristic_predict(row)

    cols = list(getattr(probability_bundle, "feature_columns", None) or [])
    X_bg, auto_cols = _numeric_frame(feat_df, cols or None)
    columns = cols if cols else auto_cols
    predict_fn = _predict_fn_from_bundle(probability_bundle)
    model = None
    if probability_bundle is not None and getattr(probability_bundle, "models", None):
        model = probability_bundle.models.get(5) or next(iter(probability_bundle.models.values()), None)

    shap_block = compute_shap(
        row,
        X_bg if not X_bg.empty else feat_df,
        columns,
        predict_fn,
        model=model,
        use_kernel_shap=use_kernel_shap,
    )
    attention_maps = compute_attention_maps(feat_df, columns, external_attention=attention)
    importance = compute_feature_importance(probability_bundle, columns, shap_block)
    base_p = float(pred["probability"])
    counterfactuals = compute_counterfactuals(row, columns, predict_fn, base_pred=base_p)
    pdp = compute_partial_dependence(row, X_bg if not X_bg.empty else feat_df, columns, predict_fn)
    confidence = compute_model_confidence(
        pred,
        row,
        shap_block,
        horizon_confidence=float(pred.get("confidence") or 0.5),
    )
    trace = build_decision_trace(row, pred, shap_block, confidence)
    schema_expl = explain_row(row, horizons or [])
    why = _why_narrative(pred, shap_block, confidence, counterfactuals)

    return ExplanationBundle(
        symbol=symbol.upper(),
        as_of=as_of_s,
        prediction=pred,
        why=why,
        shap=shap_block,
        attention_maps=attention_maps,
        feature_importance=importance,
        counterfactuals=counterfactuals,
        partial_dependence=pdp,
        decision_trace=trace,
        model_confidence=confidence,
        schema_explanations=schema_expl,
        meta={
            "has_shap_lib": HAS_SHAP,
            "n_features": len(columns),
            "predict_fn": "model" if predict_fn else "heuristic",
        },
    )


def engine_meta() -> dict[str, Any]:
    return {
        "service": "explainability",
        "version": VERSION,
        "provide": list(EXPLAIN_COMPONENTS),
        "provide_keys": {
            "SHAP": "shap",
            "Attention Maps": "attention_maps",
            "Feature Importance": "feature_importance",
            "Counterfactual Analysis": "counterfactuals",
            "Partial Dependence": "partial_dependence",
            "Decision Trace": "decision_trace",
            "Model Confidence": "model_confidence",
        },
        "has_shap": HAS_SHAP,
        "rule": "Every prediction must explain WHY it was produced",
    }
