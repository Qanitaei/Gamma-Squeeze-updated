"""Classification / probability metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from gamma_squeeze.training.validation import ranking_auc


def _binary(y: np.ndarray) -> np.ndarray:
    return np.asarray(y).astype(int).reshape(-1)


def _probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    return np.clip(p, 1e-15, 1.0 - 1e-15)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = _binary(y_true)
    pred = _binary(y_pred)
    if len(y) == 0:
        return 0.0
    return float(np.mean(y == pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = _binary(y_true)
    pred = _binary(y_pred)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    den = tp + fp
    return float(tp / den) if den > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = _binary(y_true)
    pred = _binary(y_pred)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    den = tp + fn
    return float(tp / den) if den > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    den = p + r
    return float(2.0 * p * r / den) if den > 0 else 0.0


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return ranking_auc(y_true, scores)


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = _binary(y_true).astype(float)
    p = _probs(proba)
    if len(y) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    y = _binary(y_true).astype(float)
    p = _probs(proba)
    if len(y) == 0:
        return 0.0
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def classification_metrics(
    y_true: np.ndarray,
    *,
    proba: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y = _binary(y_true)
    if proba is None and y_pred is None:
        return {"status": "missing_predictions", "n": int(len(y))}
    p = _probs(proba) if proba is not None else None
    pred = _binary(y_pred) if y_pred is not None else (p >= threshold).astype(int)
    out: dict[str, Any] = {
        "status": "ok",
        "n": int(len(y)),
        "threshold": threshold,
        "Accuracy": accuracy(y, pred),
        "Precision": precision(y, pred),
        "Recall": recall(y, pred),
        "F1": f1_score(y, pred),
    }
    if p is not None:
        out["ROC AUC"] = roc_auc(y, p)
        out["Brier Score"] = brier_score(y, p)
        out["Log Loss"] = log_loss(y, p)
    return out
