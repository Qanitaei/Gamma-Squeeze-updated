"""Drift detection — feature, prediction, and concept drift."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class DriftReport:
    feature_drift: dict[str, Any] = field(default_factory=dict)
    prediction_drift: dict[str, Any] = field(default_factory=dict)
    concept_drift: dict[str, Any] = field(default_factory=dict)
    triggered: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _psi(expected: np.ndarray, actual: np.ndarray, *, buckets: int = 10) -> float:
    """Population Stability Index between two 1d arrays."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return 0.0
    qs = np.linspace(0, 100, buckets + 1)
    bins = np.unique(np.percentile(expected, qs))
    if len(bins) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=bins)
    a_counts, _ = np.histogram(actual, bins=bins)
    e = e_counts / max(e_counts.sum(), 1)
    a = a_counts / max(a_counts.sum(), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float)[np.isfinite(a)])
    b = np.sort(np.asarray(b, dtype=float)[np.isfinite(b)])
    if len(a) < 5 or len(b) < 5:
        return 0.0
    # Approximate two-sample KS via CDF on pooled grid
    grid = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def detect_feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    psi_threshold: float = 0.25,
    max_features: int = 40,
) -> dict[str, Any]:
    """Per-feature PSI; flag if mean PSI or any feature exceeds threshold."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
        return {"status": "shape_mismatch", "triggered": False, "mean_psi": 0.0, "features": []}
    names = feature_names or [f"f{i}" for i in range(ref.shape[1])]
    rows = []
    for i in range(min(ref.shape[1], max_features)):
        psi = _psi(ref[:, i], cur[:, i])
        rows.append({"feature": names[i], "psi": psi, "drift": psi >= psi_threshold})
    rows.sort(key=lambda d: -d["psi"])
    mean_psi = float(np.mean([r["psi"] for r in rows])) if rows else 0.0
    triggered = mean_psi >= psi_threshold or any(r["drift"] for r in rows[:5])
    return {
        "status": "ok",
        "method": "PSI",
        "threshold": psi_threshold,
        "mean_psi": mean_psi,
        "triggered": triggered,
        "features": rows[:15],
    }


def detect_prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    *,
    ks_threshold: float = 0.20,
    mean_shift_threshold: float = 0.10,
) -> dict[str, Any]:
    ref = np.asarray(reference_scores, dtype=float).reshape(-1)
    cur = np.asarray(current_scores, dtype=float).reshape(-1)
    ks = _ks_stat(ref, cur)
    mean_shift = float(abs(np.nanmean(cur) - np.nanmean(ref))) if len(ref) and len(cur) else 0.0
    triggered = ks >= ks_threshold or mean_shift >= mean_shift_threshold
    return {
        "status": "ok",
        "method": "KS+mean_shift",
        "ks": ks,
        "mean_shift": mean_shift,
        "ks_threshold": ks_threshold,
        "mean_shift_threshold": mean_shift_threshold,
        "triggered": triggered,
        "reference_mean": float(np.nanmean(ref)) if len(ref) else None,
        "current_mean": float(np.nanmean(cur)) if len(cur) else None,
    }


def detect_concept_drift(
    y_reference: np.ndarray,
    scores_reference: np.ndarray,
    y_current: np.ndarray,
    scores_current: np.ndarray,
    *,
    auc_drop_threshold: float = 0.08,
    label_shift_threshold: float = 0.15,
) -> dict[str, Any]:
    """Concept drift via AUC degradation + label distribution shift."""
    from gamma_squeeze.training.validation import ranking_auc

    y_r = np.asarray(y_reference).astype(int)
    y_c = np.asarray(y_current).astype(int)
    s_r = np.asarray(scores_reference, dtype=float)
    s_c = np.asarray(scores_current, dtype=float)
    auc_r = ranking_auc(y_r, s_r) if len(y_r) else 0.5
    auc_c = ranking_auc(y_c, s_c) if len(y_c) else 0.5
    auc_drop = float(auc_r - auc_c)
    rate_r = float(np.mean(y_r)) if len(y_r) else 0.0
    rate_c = float(np.mean(y_c)) if len(y_c) else 0.0
    label_shift = abs(rate_c - rate_r)
    triggered = auc_drop >= auc_drop_threshold or label_shift >= label_shift_threshold
    return {
        "status": "ok",
        "method": "auc_drop+label_shift",
        "auc_reference": auc_r,
        "auc_current": auc_c,
        "auc_drop": auc_drop,
        "label_rate_reference": rate_r,
        "label_rate_current": rate_c,
        "label_shift": label_shift,
        "auc_drop_threshold": auc_drop_threshold,
        "label_shift_threshold": label_shift_threshold,
        "triggered": triggered,
    }


def detect_all_drift(
    *,
    X_ref: np.ndarray,
    X_cur: np.ndarray,
    scores_ref: np.ndarray,
    scores_cur: np.ndarray,
    y_ref: np.ndarray | None = None,
    y_cur: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> DriftReport:
    feat = detect_feature_drift(X_ref, X_cur, feature_names)
    pred = detect_prediction_drift(scores_ref, scores_cur)
    if y_ref is not None and y_cur is not None and len(y_ref) and len(y_cur):
        concept = detect_concept_drift(y_ref, scores_ref, y_cur, scores_cur)
    else:
        concept = {"status": "skipped", "triggered": False, "reason": "labels_unavailable"}

    reasons = []
    if feat.get("triggered"):
        reasons.append("feature_drift")
    if pred.get("triggered"):
        reasons.append("prediction_drift")
    if concept.get("triggered"):
        reasons.append("concept_drift")
    return DriftReport(
        feature_drift=feat,
        prediction_drift=pred,
        concept_drift=concept,
        triggered=bool(reasons),
        reasons=reasons,
    )
