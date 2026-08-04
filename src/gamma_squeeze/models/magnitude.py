"""Expected magnitude (forward return) via quantile regression baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gamma_squeeze.config import HORIZONS, resolve_models_root


@dataclass
class MagnitudeModelBundle:
    horizons: tuple[int, ...] = HORIZONS
    models_p50: dict[int, Any] = field(default_factory=dict)
    models_p10: dict[int, Any] = field(default_factory=dict)
    models_p90: dict[int, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    version: str = "mag-gbr-quantile-v1"

    def predict_row(self, x: pd.Series | dict[str, Any]) -> dict[int, dict[str, float]]:
        vec = self._vectorize(x)
        out: dict[int, dict[str, float]] = {}
        for h in self.horizons:
            p10 = float(self.models_p10[h].predict(vec)[0]) if h in self.models_p10 else 0.0
            p50 = float(self.models_p50[h].predict(vec)[0]) if h in self.models_p50 else 0.0
            p90 = float(self.models_p90[h].predict(vec)[0]) if h in self.models_p90 else 0.0
            # enforce order
            ordered = sorted([p10, p50, p90])
            out[h] = {"p10": ordered[0], "p50": ordered[1], "p90": ordered[2], "expected": ordered[1]}
        return out

    def _vectorize(self, x: pd.Series | dict[str, Any]) -> np.ndarray:
        data = x.to_dict() if isinstance(x, pd.Series) else x
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


def _gbr(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "reg",
                GradientBoostingRegressor(
                    loss="quantile",
                    alpha=alpha,
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.08,
                    random_state=42,
                ),
            ),
        ]
    )


def train_magnitude_models(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
) -> MagnitudeModelBundle:
    # Keep feature-side names on overlap (e.g. positioning_stress also on labels).
    merged = features.merge(labels, on=["symbol", "as_of"], how="inner", suffixes=("", "_lbl"))
    cols = [c for c in feature_columns if c in merged.columns]
    bundle = MagnitudeModelBundle(feature_columns=list(cols))
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
        y_col = f"magnitude_label_{h}d"
        if y_col not in merged.columns:
            y_col = f"fwd_ret_{h}d"
        if y_col not in merged.columns:
            continue
        y = pd.to_numeric(merged[y_col], errors="coerce")
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(y_arr)
        if int(mask.sum()) < 5:
            continue
        Xh, yh = X[mask], y_arr[mask]
        for alpha, store in ((0.1, bundle.models_p10), (0.5, bundle.models_p50), (0.9, bundle.models_p90)):
            model = _gbr(alpha)
            model.fit(Xh, yh)
            store[h] = model
    return bundle


def save_magnitude_bundle(bundle: MagnitudeModelBundle, *, root: Path | None = None) -> Path:
    path = (root or resolve_models_root()) / "magnitude" / f"{bundle.version}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    joblib.dump(bundle, path.parent / "latest.joblib")
    return path


def load_magnitude_bundle(*, root: Path | None = None) -> MagnitudeModelBundle | None:
    path = (root or resolve_models_root()) / "magnitude" / "latest.joblib"
    if not path.is_file():
        return None
    return joblib.load(path)
