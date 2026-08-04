"""Multi-horizon squeeze probability with logistic + isotonic calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gamma_squeeze.config import HORIZONS, resolve_models_root


@dataclass
class ProbabilityModelBundle:
    horizons: tuple[int, ...] = HORIZONS
    models: dict[int, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    version: str = "prob-logit-isotonic-v1"

    def predict_proba_row(self, x: pd.Series | dict[str, Any]) -> dict[int, float]:
        vec = self._vectorize(x)
        out: dict[int, float] = {}
        for h, model in self.models.items():
            if model is None:
                out[h] = 0.0
                continue
            proba = model.predict_proba(vec)[0]
            # class 1 probability if binary
            out[h] = float(proba[1]) if len(proba) > 1 else float(proba[0])
        return out

    def _vectorize(self, x: pd.Series | dict[str, Any]) -> np.ndarray:
        if isinstance(x, pd.Series):
            data = x.to_dict()
        else:
            data = x
        vals: list[float] = []
        for c in self.feature_columns:
            try:
                v = float(data.get(c))
            except (TypeError, ValueError):
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            vals.append(v)
        return np.asarray([vals], dtype=float)


def train_probability_models(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
) -> ProbabilityModelBundle:
    merged = features.merge(labels, on=["symbol", "as_of"], how="inner", suffixes=("", "_lbl"))
    cols = [c for c in feature_columns if c in merged.columns]
    bundle = ProbabilityModelBundle(feature_columns=list(cols))
    if merged.empty or not cols:
        return bundle

    X = (
        merged[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy()
    )
    for h in HORIZONS:
        y_col = f"squeeze_{h}d"
        if y_col not in merged.columns:
            continue
        y = merged[y_col].fillna(0).astype(int).to_numpy()
        if len(set(y.tolist())) < 2:
            # degenerate — constant prior
            bundle.models[h] = _ConstantProba(float(y.mean()) if len(y) else 0.0)
            continue
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=500, class_weight="balanced"),
                ),
            ]
        )
        # isotonic needs enough samples; fall back to uncalibrated
        if len(y) >= 30:
            model: Any = CalibratedClassifierCV(base, method="isotonic", cv=min(3, len(y) // 10 or 2))
        else:
            model = base
        model.fit(X, y)
        bundle.models[h] = model
    return bundle


def save_probability_bundle(bundle: ProbabilityModelBundle, *, root: Path | None = None) -> Path:
    path = (root or resolve_models_root()) / "probability" / f"{bundle.version}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    latest = path.parent / "latest.joblib"
    joblib.dump(bundle, latest)
    return path


def load_probability_bundle(*, root: Path | None = None) -> ProbabilityModelBundle | None:
    path = (root or resolve_models_root()) / "probability" / "latest.joblib"
    if not path.is_file():
        return None
    return joblib.load(path)


class _ConstantProba:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])
