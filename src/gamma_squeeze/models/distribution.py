"""Return distribution from quantile heads + empirical residual bins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gamma_squeeze.config import HORIZONS
from gamma_squeeze.serve.schemas import DistributionBin


@dataclass
class DistributionModelBundle:
    horizons: tuple[int, ...] = HORIZONS
    residual_std: dict[int, float] = field(default_factory=dict)
    version: str = "dist-quantile-empirical-v1"

    def predict_bins(
        self,
        *,
        horizon: int,
        p10: float,
        p50: float,
        p90: float,
        n_bins: int = 6,
    ) -> list[DistributionBin]:
        """Construct a discrete distribution spanning p10..p90 with gaussian-ish mass."""
        lo, hi = float(p10), float(p90)
        if hi <= lo:
            hi = lo + 0.01
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        sigma = self.residual_std.get(horizon) or max((hi - lo) / 2.5, 1e-4)
        weights = np.exp(-0.5 * ((centers - p50) / sigma) ** 2)
        weights = weights / weights.sum()
        return [
            DistributionBin(
                lo_pct=float(edges[i] * 100.0),
                hi_pct=float(edges[i + 1] * 100.0),
                probability=float(weights[i]),
            )
            for i in range(n_bins)
        ]


def fit_distribution_residuals(
    labels: pd.DataFrame,
    magnitude_preds: dict[int, np.ndarray] | None = None,
) -> DistributionModelBundle:
    """Estimate residual std per horizon from labeled forward returns."""
    bundle = DistributionModelBundle()
    for h in HORIZONS:
        col = f"fwd_ret_{h}d"
        if col not in labels.columns:
            continue
        y = pd.to_numeric(labels[col], errors="coerce").dropna()
        if y.empty:
            continue
        if magnitude_preds and h in magnitude_preds and len(magnitude_preds[h]) == len(labels):
            resid = y.to_numpy() - magnitude_preds[h][: len(y)]
            bundle.residual_std[h] = float(np.nanstd(resid))
        else:
            bundle.residual_std[h] = float(y.std() or 0.02)
    return bundle
