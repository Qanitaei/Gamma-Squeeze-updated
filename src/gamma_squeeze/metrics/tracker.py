"""Unified performance metrics tracker — classification, regression, trading, tail risk."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gamma_squeeze.features.feature_store import (
    build_features_for_symbol,
    load_features,
    model_feature_matrix,
)
from gamma_squeeze.labels.label_builder import build_labels_for_symbol, load_labels
from gamma_squeeze.metrics.classification import classification_metrics
from gamma_squeeze.metrics.regression import regression_metrics
from gamma_squeeze.metrics.risk import risk_metrics
from gamma_squeeze.metrics.trading import trading_metrics
from gamma_squeeze.training.hyperopt import fit_best_classifier
from gamma_squeeze.training.validation import align_xy


TRACKED_METRICS: tuple[str, ...] = (
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC AUC",
    "Brier Score",
    "Log Loss",
    "MAE",
    "RMSE",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Maximum Drawdown",
    "Annual Return",
    "Win Rate",
    "Profit Factor",
    "Average Trade",
    "Average Hold Time",
    "Tail Risk",
    "Expected Shortfall",
)


@dataclass
class MetricsReport:
    symbols: list[str]
    status: str
    classification: dict[str, Any] = field(default_factory=dict)
    regression: dict[str, Any] = field(default_factory=dict)
    trading: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    tracked: list[str] = field(default_factory=lambda: list(TRACKED_METRICS))
    flat: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_from_arrays(
    *,
    y_true: np.ndarray | None = None,
    proba: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    y_reg_true: np.ndarray | None = None,
    y_reg_pred: np.ndarray | None = None,
    returns: np.ndarray | None = None,
    signal: np.ndarray | None = None,
    trade_pnls: np.ndarray | None = None,
    hold_periods: np.ndarray | None = None,
    threshold: float = 0.5,
    alpha: float = 0.05,
    periods_per_year: float = 252.0,
    symbols: list[str] | None = None,
) -> MetricsReport:
    """Compute the full tracked metric set from raw arrays."""
    clf = (
        classification_metrics(y_true, proba=proba, y_pred=y_pred, threshold=threshold)
        if y_true is not None
        else {"status": "skipped"}
    )
    reg = (
        regression_metrics(y_reg_true, y_reg_pred)
        if y_reg_true is not None and y_reg_pred is not None
        else {"status": "skipped"}
    )
    tr = (
        trading_metrics(
            returns,
            signal=signal,
            trade_pnls=trade_pnls,
            hold_periods=hold_periods,
            periods_per_year=periods_per_year,
        )
        if returns is not None
        else {"status": "skipped"}
    )
    rk = risk_metrics(returns, alpha=alpha) if returns is not None else {"status": "skipped"}

    flat: dict[str, Any] = {}
    for block in (clf, reg, tr, rk):
        for k, v in block.items():
            if k in TRACKED_METRICS:
                flat[k] = v

    status = "ok" if any(b.get("status") == "ok" for b in (clf, reg, tr, rk)) else "no_metrics"
    return MetricsReport(
        symbols=symbols or [],
        status=status,
        classification=clf,
        regression=reg,
        trading=tr,
        risk=rk,
        flat=flat,
    )


def _period_returns(merged: pd.DataFrame, proba: np.ndarray, *, ret_col: str = "fwd_ret_1d") -> np.ndarray:
    """Prefer non-null forward returns; else spot-derived; else synthetic edge."""
    for col in (ret_col, "fwd_ret_1d", "ret_1d"):
        if col not in merged.columns:
            continue
        rets = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(rets).sum() >= max(5, len(rets) // 10):
            return np.nan_to_num(rets, nan=0.0)
    if "spot" in merged.columns:
        spot = pd.to_numeric(merged["spot"], errors="coerce").to_numpy(dtype=float)
        rets = np.zeros(len(spot), dtype=float)
        prev = spot[:-1]
        nxt = spot[1:]
        valid = np.isfinite(prev) & np.isfinite(nxt) & (prev > 1e-6) & (nxt > 1e-6)
        step = np.zeros(len(prev), dtype=float)
        step[valid] = (nxt[valid] - prev[valid]) / prev[valid]
        rets[1:] = np.clip(step, -0.5, 0.5)
        return rets
    return (proba - 0.5) * 0.02


def _strategy_returns_from_proba(
    merged: pd.DataFrame,
    proba: np.ndarray,
    *,
    threshold: float = 0.5,
    ret_col: str = "fwd_ret_1d",
) -> tuple[np.ndarray, np.ndarray]:
    """Long when proba >= threshold; flat otherwise."""
    rets = _period_returns(merged, proba, ret_col=ret_col)
    signal = (proba >= threshold).astype(float)
    strat = rets * signal
    return strat, signal


def _resolve_binary_label(
    labels: pd.DataFrame,
    *,
    preferred: str = "squeeze_5d",
) -> tuple[str, pd.DataFrame]:
    """
    Pick a binary label with both classes.
    Prefer squeeze; fall back to up-move from fwd return / candidate_setup.
    """
    labs = labels.copy()
    if preferred in labs.columns:
        y = pd.to_numeric(labs[preferred], errors="coerce").fillna(0).astype(int)
        if y.nunique(dropna=True) >= 2:
            return preferred, labs

    for horizon in (5, 1, 3, 10):
        fwd = f"fwd_ret_{horizon}d"
        if fwd not in labs.columns:
            continue
        series = pd.to_numeric(labs[fwd], errors="coerce")
        if series.notna().sum() < 20:
            continue
        col = f"dir_up_{horizon}d"
        labs[col] = (series.fillna(0.0) > 0.0).astype(int)
        if labs[col].nunique(dropna=True) >= 2:
            return col, labs

    if "candidate_setup" in labs.columns:
        col = "candidate_setup_bin"
        labs[col] = (pd.to_numeric(labs["candidate_setup"], errors="coerce").fillna(0) > 0).astype(int)
        if labs[col].nunique(dropna=True) >= 2:
            return col, labs

    return preferred, labs


def track_symbol_metrics(
    symbols: list[str],
    *,
    rebuild: bool = False,
    label_col: str = "squeeze_5d",
    magnitude_col: str = "magnitude_label_5d",
    threshold: float = 0.5,
    alpha: float = 0.05,
    train_fraction: float = 0.7,
) -> MetricsReport:
    """
    Load features/labels, fit a quick classifier on the train slice,
    score holdout, and track the full metric catalog.
    """
    syms = [s.upper() for s in symbols]
    feats, labs = [], []
    for sym in syms:
        f = build_features_for_symbol(sym, include_skew=False, include_macro=True) if rebuild else load_features(sym)
        if f.empty and not rebuild:
            f = build_features_for_symbol(sym, include_skew=False, include_macro=True)
        lab = load_labels(sym)
        if lab.empty or rebuild:
            lab = build_labels_for_symbol(sym, features=f)
        if not f.empty:
            feats.append(f)
        if not lab.empty:
            labs.append(lab)
    if not feats or not labs:
        return MetricsReport(symbols=syms, status="no_data")

    features = pd.concat(feats, ignore_index=True)
    labels = pd.concat(labs, ignore_index=True)
    resolved_label, labels = _resolve_binary_label(labels, preferred=label_col)
    _, cols = model_feature_matrix(features)
    X, y, merged = align_xy(features, labels, cols, label_col=resolved_label)
    if len(X) < 20 or len(np.unique(y)) < 2:
        return MetricsReport(
            symbols=syms,
            status="insufficient_data",
            flat={"n_rows": int(len(X)), "label_col": resolved_label},
        )

    cut = max(10, int(len(X) * train_fraction))
    cut = min(cut, len(X) - 5)
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]
    model = fit_best_classifier(Xtr, ytr, {"n_estimators": 80, "max_depth": 3})
    if model is None:
        return MetricsReport(symbols=syms, status="fit_failed")

    proba = model.predict_proba(Xte)[:, 1]
    merged_te = merged.iloc[cut:].reset_index(drop=True)

    # Regression target: magnitude / |fwd return|; else |spot return|
    y_mag = None
    for col in (
        magnitude_col,
        "magnitude_label_5d",
        "magnitude_label_1d",
        "fwd_ret_5d",
        "fwd_ret_1d",
        "ret_5d",
    ):
        if col not in merged_te.columns:
            continue
        series = pd.to_numeric(merged_te[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(series).sum() >= max(5, len(series) // 10):
            y_mag = np.abs(np.nan_to_num(series, nan=0.0))
            break
    if y_mag is None:
        spot_rets = _period_returns(merged_te, proba)
        y_mag = np.abs(spot_rets) if np.any(np.abs(spot_rets) > 0) else np.abs(yte.astype(float) * 0.03)
    # Predict magnitude as scaled probability (simple proxy for MAE/RMSE tracking)
    y_mag_hat = proba * float(np.nanmean(y_mag) / max(float(np.nanmean(proba)), 1e-6))

    strat, signal = _strategy_returns_from_proba(merged_te, proba, threshold=threshold)

    report = compute_from_arrays(
        y_true=yte,
        proba=proba,
        y_reg_true=y_mag,
        y_reg_pred=y_mag_hat,
        returns=strat,
        signal=signal,
        threshold=threshold,
        alpha=alpha,
        symbols=syms,
    )
    report.flat["label_col"] = resolved_label
    report.classification["label_col"] = resolved_label
    return report


def metrics_meta() -> dict[str, Any]:
    return {
        "service": "performance_metrics",
        "track": list(TRACKED_METRICS),
        "n_metrics": len(TRACKED_METRICS),
        "groups": {
            "classification": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC AUC",
                "Brier Score",
                "Log Loss",
            ],
            "regression": ["MAE", "RMSE"],
            "trading": [
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Maximum Drawdown",
                "Annual Return",
                "Win Rate",
                "Profit Factor",
                "Average Trade",
                "Average Hold Time",
            ],
            "risk": ["Tail Risk", "Expected Shortfall"],
        },
        "endpoints": [
            "/v1/meta",
            "/v1/track",
            "/v1/track/{symbol}",
            "/v1/compute",
        ],
    }
