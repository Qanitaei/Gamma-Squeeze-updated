"""Walk-forward validation and time-series splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class FoldResult:
    fold: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    method: str
    n_splits: int
    folds: list[FoldResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_splits": self.n_splits,
            "folds": [f.to_dict() for f in self.folds],
            "aggregate": self.aggregate,
        }


def time_series_splits(
    n_samples: int,
    *,
    n_splits: int = 5,
    gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """sklearn TimeSeriesSplit indices."""
    if n_samples < n_splits + 2:
        n_splits = max(2, min(n_splits, max(2, n_samples // 3)))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    X_dummy = np.zeros((n_samples, 1))
    return list(splitter.split(X_dummy))


def walk_forward_splits(
    n_samples: int,
    *,
    n_splits: int = 5,
    min_train_size: int | None = None,
    test_size: int | None = None,
    expanding: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Anchored (expanding) or rolling walk-forward windows."""
    if n_samples < 10:
        return time_series_splits(n_samples, n_splits=max(2, min(n_splits, 2)))

    test_size = test_size or max(1, n_samples // (n_splits + 1))
    min_train = min_train_size or max(test_size * 2, n_samples // (n_splits + 2))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    # Place test windows from the end backwards, then reverse
    cursor = n_samples
    for _ in range(n_splits):
        test_end = cursor
        test_start = max(min_train, test_end - test_size)
        if test_start >= test_end:
            break
        if expanding:
            train_start = 0
        else:
            train_start = max(0, test_start - min_train)
        if test_start - train_start < max(5, min_train // 2):
            break
        train_idx = np.arange(train_start, test_start)
        test_idx = np.arange(test_start, test_end)
        folds.append((train_idx, test_idx))
        cursor = test_start
        if cursor <= min_train:
            break
    folds.reverse()
    return folds or time_series_splits(n_samples, n_splits=max(2, min(n_splits, 3)))


def iter_splits(
    n_samples: int,
    *,
    method: str = "walk_forward",
    n_splits: int = 5,
    gap: int = 0,
    expanding: bool = True,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    if method == "time_series_split":
        splits = time_series_splits(n_samples, n_splits=n_splits, gap=gap)
    else:
        splits = walk_forward_splits(n_samples, n_splits=n_splits, expanding=expanding)
    for i, (tr, te) in enumerate(splits):
        yield i, tr, te


def ranking_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = sum(float((neg < p).mean()) for p in pos)
    return float(wins / len(pos))


def evaluate_splits(
    X: np.ndarray,
    y: np.ndarray,
    fit_predict: Any,
    *,
    method: str = "walk_forward",
    n_splits: int = 5,
    expanding: bool = True,
) -> ValidationReport:
    """
    fit_predict(X_train, y_train, X_test) -> scores array for X_test
    """
    folds: list[FoldResult] = []
    aucs: list[float] = []
    for i, tr, te in iter_splits(len(X), method=method, n_splits=n_splits, expanding=expanding):
        scores = np.asarray(fit_predict(X[tr], y[tr], X[te]), dtype=float).reshape(-1)
        auc = ranking_auc(y[te], scores)
        aucs.append(auc)
        folds.append(
            FoldResult(
                fold=i,
                train_end=int(tr[-1]) if len(tr) else 0,
                test_start=int(te[0]) if len(te) else 0,
                test_end=int(te[-1]) if len(te) else 0,
                n_train=int(len(tr)),
                n_test=int(len(te)),
                metrics={"auc": auc, "pos_rate": float(np.mean(y[te])) if len(te) else 0.0},
            )
        )
    agg = {
        "auc_mean": float(np.mean(aucs)) if aucs else 0.5,
        "auc_std": float(np.std(aucs)) if len(aucs) > 1 else 0.0,
        "n_folds": float(len(folds)),
    }
    return ValidationReport(method=method, n_splits=len(folds), folds=folds, aggregate=agg)


def align_xy(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
    *,
    label_col: str = "squeeze_5d",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    merged = features.merge(labels, on=["symbol", "as_of"], how="inner")
    if merged.empty or label_col not in merged.columns:
        return np.zeros((0, len(feature_columns))), np.zeros(0), merged
    merged = merged.sort_values("as_of")
    cols = [c for c in feature_columns if c in merged.columns]
    X = merged[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = merged[label_col].fillna(0).astype(int).to_numpy()
    return X, y, merged
