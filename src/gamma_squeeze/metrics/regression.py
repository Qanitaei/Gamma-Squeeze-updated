"""Regression error metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(yt), len(yp))
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(yt[:n] - yp[:n])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(yt), len(yp))
    if n == 0:
        return 0.0
    return float(np.sqrt(np.mean((yt[:n] - yp[:n]) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(yt), len(yp))
    return {
        "status": "ok" if n else "empty",
        "n": int(n),
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
    }
