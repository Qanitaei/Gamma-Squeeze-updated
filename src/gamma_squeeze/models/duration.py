"""Expected squeeze duration (trading days) baseline."""

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
class DurationModelBundle:
    horizons: tuple[int, ...] = HORIZONS
    models: dict[int, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    version: str = "dur-gbr-v1"

    def predict_row(self, x: pd.Series | dict[str, Any]) -> dict[int, float]:
        vec = self._vectorize(x)
        out: dict[int, float] = {}
        for h, model in self.models.items():
            pred = float(model.predict(vec)[0])
            out[h] = float(np.clip(pred, 1.0, float(h)))
        for h in self.horizons:
            out.setdefault(h, float(h))
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


def train_duration_models(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
) -> DurationModelBundle:
    # Keep feature-side names on overlap (e.g. positioning_stress also on labels).
    merged = features.merge(labels, on=["symbol", "as_of"], how="inner", suffixes=("", "_lbl"))
    cols = [c for c in feature_columns if c in merged.columns]
    bundle = DurationModelBundle(feature_columns=list(cols))
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
        y_col = f"duration_label_{h}d"
        if y_col not in merged.columns:
            continue
        y = pd.to_numeric(merged[y_col], errors="coerce")
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(y_arr)
        if int(mask.sum()) < 5:
            continue
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "reg",
                    GradientBoostingRegressor(
                        n_estimators=80,
                        max_depth=3,
                        learning_rate=0.08,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X[mask], y_arr[mask])
        bundle.models[h] = model
    return bundle


def save_duration_bundle(bundle: DurationModelBundle, *, root: Path | None = None) -> Path:
    path = (root or resolve_models_root()) / "duration" / f"{bundle.version}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    joblib.dump(bundle, path.parent / "latest.joblib")
    return path


def load_duration_bundle(*, root: Path | None = None) -> DurationModelBundle | None:
    path = (root or resolve_models_root()) / "duration" / "latest.joblib"
    if not path.is_file():
        return None
    return joblib.load(path)
