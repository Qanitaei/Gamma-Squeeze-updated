"""Temporal Fusion Transformer–style multi-target quantile forecasting.

Primary path: lightweight temporal-attention + sklearn quantile GBTs
(works without PyTorch). Optional backend hook for pytorch-forecasting when installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from gamma_squeeze.config import resolve_models_root
from gamma_squeeze.retrain.registry import register_backend

TFT_HORIZONS = (1, 2, 3, 5, 10)
TFT_HORIZON_LABELS = {
    1: "1 Trading Day",
    2: "2 Trading Days",
    3: "3 Trading Days",
    5: "5 Trading Days",
    10: "10 Trading Days",
}
TFT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
TFT_TARGETS = (
    "future_price",
    "iv",
    "net_gex",
    "dealer_hedge_requirement",
    "gamma_flip",
    "call_wall",
    "put_wall",
    "expected_move",
)
TFT_TARGET_LABELS = {
    "future_price": "Future Price",
    "iv": "IV",
    "net_gex": "Net GEX",
    "dealer_hedge_requirement": "Dealer Hedge Requirement",
    "gamma_flip": "Gamma Flip",
    "call_wall": "Call Wall",
    "put_wall": "Put Wall",
    "expected_move": "Expected Move",
}
TFT_OUTPUTS = (
    "median_forecast",  # q50
    "q10",
    "q25",
    "q75",
    "q90",
    "prediction_interval",  # q10–q90 (+ mid q25–q75)
    "attention_weights",  # temporal + variable
)

# Feature-store columns (ordered candidates) feeding each forecast target
TARGET_SOURCE_COLUMNS: dict[str, list[str]] = {
    "future_price": ["spot"],
    "iv": ["atm_iv", "iv_atm", "historical_volatility", "realized_volatility"],
    "net_gex": ["net_gex", "gamma_exposure", "gex_per_spot"],
    "dealer_hedge_requirement": ["dealer_hedge_requirement", "dealer_delta"],
    "gamma_flip": ["gamma_flip", "zero_gamma"],
    "call_wall": ["call_wall"],
    "put_wall": ["put_wall"],
    "expected_move": ["expected_move", "expected_move_pct"],
}

COVARIATE_CANDIDATES = [
    "spot",
    "ret_1d",
    "ret_5d",
    "rvol_10d",
    "net_gex",
    "gex_per_spot",
    "gamma_flip",
    "zero_gamma",
    "call_wall",
    "put_wall",
    "dealer_hedge_requirement",
    "dealer_gamma",
    "dealer_delta",
    "positioning_stress",
    "pcr_oi",
    "atm_iv",
    "iv_atm",
    "expected_move",
    "expected_move_pct",
    "rsi",
    "atr",
    "vix",
    "yield_spread",
    "fed_funds",
]


def _resolve_target_series(df: pd.DataFrame, target: str) -> pd.Series:
    for col in TARGET_SOURCE_COLUMNS.get(target, []):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s
    # derived fallbacks
    if target == "future_price" and "spot" in df.columns:
        return pd.to_numeric(df["spot"], errors="coerce")
    if target == "expected_move":
        spot = pd.to_numeric(df.get("spot"), errors="coerce") if "spot" in df.columns else None
        iv = None
        for c in ("atm_iv", "iv_atm", "rvol_10d"):
            if c in df.columns:
                iv = pd.to_numeric(df[c], errors="coerce")
                break
        if spot is not None and iv is not None:
            # 1σ 5d move proxy when expected_move missing
            return spot * iv * np.sqrt(5 / 252.0)
    return pd.Series(np.nan, index=df.index)


def _pick_covariates(df: pd.DataFrame, feature_columns: list[str] | None = None) -> list[str]:
    preferred = list(feature_columns or [])
    cols: list[str] = []
    for c in COVARIATE_CANDIDATES + preferred:
        if c in df.columns and c not in cols and c not in ("symbol", "as_of"):
            cols.append(c)
    # ensure target sources present
    for sources in TARGET_SOURCE_COLUMNS.values():
        for c in sources:
            if c in df.columns and c not in cols:
                cols.append(c)
    return cols[:48]


@dataclass
class TemporalAttention:
    """Learned temporal + variable attention over a lookback window."""

    lookback: int = 32
    feature_columns: list[str] = field(default_factory=list)
    temporal_logits: np.ndarray | None = None  # (L,)
    variable_logits: np.ndarray | None = None  # (F,)

    def fit(self, windows: np.ndarray, y: np.ndarray) -> "TemporalAttention":
        """windows: (N, L, F). Fit attention via absolute correlation with target."""
        n, l, f = windows.shape
        self.lookback = l
        # temporal: which lag correlates with y (using mean across features)
        temp_scores = np.zeros(l, dtype=float)
        for t in range(l):
            pooled = windows[:, t, :].mean(axis=1)
            temp_scores[t] = abs(_safe_corr(pooled, y))
        self.temporal_logits = temp_scores
        # variable: which feature (attention-pooled over time with uniform) correlates
        var_scores = np.zeros(f, dtype=float)
        for j in range(f):
            pooled = windows[:, :, j].mean(axis=1)
            var_scores[j] = abs(_safe_corr(pooled, y))
        self.variable_logits = var_scores
        return self

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        t = _softmax(self.temporal_logits if self.temporal_logits is not None else np.ones(self.lookback))
        v = _softmax(
            self.variable_logits
            if self.variable_logits is not None
            else np.ones(max(1, len(self.feature_columns)))
        )
        return t, v

    def transform(self, window: np.ndarray) -> np.ndarray:
        """window (L, F) → context vector (F,) using temporal attention, then var-gate."""
        t_w, v_w = self.weights()
        l = min(len(t_w), window.shape[0])
        f = min(len(v_w), window.shape[1])
        ctx = (t_w[-l:, None] * window[-l:, :f]).sum(axis=0)
        return ctx * v_w[:f]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - np.max(z)
    e = np.exp(np.clip(z, -50, 50))
    return e / max(e.sum(), 1e-12)


def _predict_quantiles(
    q_models: dict[Any, Any],
    x: np.ndarray,
    quantiles: list[float],
) -> dict[float, float]:
    """Predict requested quantiles; support median_spread sigma packing."""
    out: dict[float, float] = {}
    sigma = q_models.get("_sigma")
    z = {0.10: -1.2816, 0.25: -0.6745, 0.50: 0.0, 0.75: 0.6745, 0.90: 1.2816}
    if sigma is not None and 0.50 in q_models:
        med = float(q_models[0.50].predict(x)[0])
        for q in quantiles:
            out[float(q)] = med + float(sigma) * z.get(float(q), 0.0)
        return out
    for q in quantiles:
        model = q_models.get(float(q))
        if model is None:
            continue
        out[float(q)] = float(model.predict(x)[0])
    # fill missing via order stats on available
    if 0.50 not in out and out:
        out[0.50] = float(np.median(list(out.values())))
    return out


def _make_quantile_regressor(q: float) -> Any:
    # Prefer HistGBM quantile when available
    try:
        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_depth=4,
            learning_rate=0.08,
            max_iter=80,
            random_state=42,
        )
    except TypeError:
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            max_depth=3,
            learning_rate=0.08,
            n_estimators=80,
            random_state=42,
        )


@dataclass
class TFTForecastModel:
    """TFT-style multi-horizon / multi-target quantile forecaster."""

    lookback: int = 32
    horizons: list[int] = field(default_factory=lambda: list(TFT_HORIZONS))
    targets: list[str] = field(default_factory=lambda: list(TFT_TARGETS))
    quantiles: list[float] = field(default_factory=lambda: list(TFT_QUANTILES))
    feature_columns: list[str] = field(default_factory=list)
    attention: TemporalAttention | None = None
    # models[target][horizon][quantile] -> regressor
    models: dict[str, dict[int, dict[float, Any]]] = field(default_factory=dict)
    version: str = "tft-quantile-v1"
    backend: str = "sklearn-tft-style"
    train_rows: int = 0
    # "full" fits every quantile; "median_spread" fits q50 then scales outer quantiles from residual σ
    quantile_mode: str = "full"

    def fit(self, features: pd.DataFrame, labels: pd.DataFrame | None = None) -> dict[str, Any]:
        del labels  # targets are built from feature panel levels
        df = features.sort_values("as_of").reset_index(drop=True) if "as_of" in features.columns else features.copy()
        self.feature_columns = _pick_covariates(df, self.feature_columns)
        if len(df) < self.lookback + max(self.horizons) + 5 or not self.feature_columns:
            return {"status": "insufficient_data", "n_rows": int(len(df)), "version": self.version}

        X_panel = (
            df[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        target_series = {t: _resolve_target_series(df, t).ffill().to_numpy(dtype=float) for t in self.targets}

        # Build windows
        windows: list[np.ndarray] = []
        indices: list[int] = []
        max_h = max(self.horizons)
        for i in range(self.lookback - 1, len(df) - max_h):
            windows.append(X_panel[i - self.lookback + 1 : i + 1])
            indices.append(i)
        if len(windows) < 20:
            return {"status": "insufficient_windows", "n_windows": len(windows), "version": self.version}

        W = np.stack(windows, axis=0)  # (N, L, F)
        # Fit attention on primary target (future_price @ horizon 5 or first)
        primary = "future_price" if "future_price" in target_series else self.targets[0]
        h_att = 5 if 5 in self.horizons else self.horizons[0]
        y_att = np.asarray([target_series[primary][i + h_att] for i in indices], dtype=float)
        self.attention = TemporalAttention(lookback=self.lookback, feature_columns=self.feature_columns)
        self.attention.fit(W, y_att)

        # Context features for tabular quantile models
        X_ctx = np.stack([self.attention.transform(w) for w in W], axis=0)
        # append last raw values + simple momentum
        last = W[:, -1, :]
        mom = W[:, -1, :] - W[:, 0, :]
        X_design = np.concatenate([X_ctx, last, mom], axis=1)

        self.models = {}
        fitted = 0
        for target in self.targets:
            series = target_series[target]
            if not np.isfinite(series).any():
                continue
            self.models[target] = {}
            for h in self.horizons:
                y = np.asarray([series[i + h] for i in indices], dtype=float)
                mask = np.isfinite(y) & np.isfinite(X_design).all(axis=1)
                if mask.sum() < 20:
                    continue
                self.models[target][h] = {}
                if self.quantile_mode == "median_spread":
                    med_model = _make_quantile_regressor(0.50)
                    med_model.fit(X_design[mask], y[mask])
                    resid = y[mask] - med_model.predict(X_design[mask])
                    sigma = float(np.std(resid) + 1e-9)
                    # store median model under q50; spreads applied at predict via meta
                    self.models[target][h][0.50] = med_model
                    self.models[target][h]["_sigma"] = sigma  # type: ignore[assignment]
                    fitted += 1
                else:
                    for q in self.quantiles:
                        model = _make_quantile_regressor(float(q))
                        model.fit(X_design[mask], y[mask])
                        self.models[target][h][float(q)] = model
                        fitted += 1

        self.train_rows = int(len(indices))
        return {
            "status": "ok",
            "n_rows": int(len(df)),
            "n_windows": int(len(indices)),
            "n_models": fitted,
            "lookback": self.lookback,
            "horizons": list(self.horizons),
            "targets": list(self.targets),
            "quantiles": list(self.quantiles),
            "quantile_mode": self.quantile_mode,
            "backend": self.backend,
            "version": self.version,
            "n_features": len(self.feature_columns),
        }

    def _encode_latest(self, features: pd.DataFrame) -> tuple[np.ndarray | None, dict[str, Any]]:
        df = features.sort_values("as_of") if "as_of" in features.columns else features
        cols = self.feature_columns or _pick_covariates(df)
        if len(df) < self.lookback or not cols:
            return None, {}
        panel = (
            df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
            .fillna(0.0)
            .tail(self.lookback)
            .to_numpy(dtype=float)
        )
        if self.attention is None:
            self.attention = TemporalAttention(lookback=self.lookback, feature_columns=cols)
            # uniform attention if unfitted
            self.attention.temporal_logits = np.linspace(0.5, 1.0, self.lookback)
            self.attention.variable_logits = np.ones(len(cols))
        ctx = self.attention.transform(panel)
        last = panel[-1]
        mom = panel[-1] - panel[0]
        x = np.concatenate([ctx, last, mom]).reshape(1, -1)
        t_w, v_w = self.attention.weights()
        attention = {
            "temporal": [
                {"lag": int(self.lookback - 1 - i), "weight": float(t_w[i])}
                for i in range(len(t_w))
            ],
            "variables": [
                {"feature": cols[j], "weight": float(v_w[j])}
                for j in range(min(len(cols), len(v_w)))
            ],
        }
        # sort variables by weight desc for readability
        attention["variables"] = sorted(attention["variables"], key=lambda r: -r["weight"])
        return x, attention

    def forecast(self, features: pd.DataFrame) -> dict[str, Any]:
        x, attention = self._encode_latest(features)
        if x is None:
            return {
                "horizons": {},
                "attention_weights": {},
                "version": self.version,
                "backend": self.backend,
                "error": "insufficient_history",
            }

        # Heuristic fallback if untrained
        if not self.models:
            return self._heuristic_forecast(features, attention)

        horizons_out: dict[str, Any] = {}
        for h in self.horizons:
            block: dict[str, Any] = {}
            for target in self.targets:
                q_models = (self.models.get(target) or {}).get(h) or {}
                if not q_models:
                    continue
                qs = _predict_quantiles(q_models, x, self.quantiles)
                q10, q25, median, q75, q90 = qs[0.10], qs[0.25], qs[0.50], qs[0.75], qs[0.90]
                ordered = [q10, q25, median, q75, q90]
                if all(v is not None for v in ordered):
                    arr = np.sort(np.asarray(ordered, dtype=float))
                    q10, q25, median, q75, q90 = [float(v) for v in arr]
                block[target] = {
                    "label": TFT_TARGET_LABELS.get(target, target),
                    "median": median,
                    "median_forecast": median,
                    "q10": q10,
                    "q25": q25,
                    "q75": q75,
                    "q90": q90,
                    "p10": q10,
                    "p25": q25,
                    "p50": median,
                    "p75": q75,
                    "p90": q90,
                    "prediction_interval": {
                        "low": q10,
                        "high": q90,
                        "mid_low": q25,
                        "mid_high": q75,
                        "coverage": "80pct_q10_q90",
                    },
                }
            horizons_out[f"{h}d"] = {
                "horizon": h,
                "horizon_label": TFT_HORIZON_LABELS.get(h, f"{h} Trading Days"),
                "targets": block,
                **block,  # flat target keys for convenience
            }

        return {
            "horizons": horizons_out,
            "attention_weights": attention,
            "version": self.version,
            "backend": self.backend,
            "lookback": self.lookback,
            "targets": list(self.targets),
            "target_labels": {t: TFT_TARGET_LABELS.get(t, t) for t in self.targets},
            "horizon_list": list(self.horizons),
            "horizon_labels": {str(h): TFT_HORIZON_LABELS.get(h, f"{h}d") for h in self.horizons},
            "quantiles": list(self.quantiles),
            "outputs": list(TFT_OUTPUTS),
        }

    def _heuristic_forecast(self, features: pd.DataFrame, attention: dict[str, Any]) -> dict[str, Any]:
        """Drift + vol scaling when models are not yet fit."""
        df = features.sort_values("as_of") if "as_of" in features.columns else features
        last = df.iloc[-1]
        ret = float(pd.to_numeric(df.get("ret_1d"), errors="coerce").tail(5).mean() or 0.0) if "ret_1d" in df.columns else 0.0
        vol = float(last.get("rvol_10d") or last.get("atr") or 0.2)
        spot = float(last.get("spot") or 0.0)
        horizons_out: dict[str, Any] = {}
        for h in self.horizons:
            scale = np.sqrt(h / 252.0) * max(vol, 1e-6)
            block: dict[str, Any] = {}
            for target in self.targets:
                series = _resolve_target_series(df, target)
                level = float(series.iloc[-1]) if series.notna().any() else 0.0
                if target == "future_price":
                    med = spot * (1.0 + ret * h)
                    spread = spot * scale
                else:
                    med = level * (1.0 + 0.15 * ret * h)
                    spread = abs(level) * scale * 0.5 + (1.0 + abs(level) * 0.01) * 0.01 * h
                q10, q25, q75, q90 = med - 1.28 * spread, med - 0.67 * spread, med + 0.67 * spread, med + 1.28 * spread
                block[target] = {
                    "label": TFT_TARGET_LABELS.get(target, target),
                    "median": med,
                    "median_forecast": med,
                    "q10": q10,
                    "q25": q25,
                    "q75": q75,
                    "q90": q90,
                    "p10": q10,
                    "p25": q25,
                    "p50": med,
                    "p75": q75,
                    "p90": q90,
                    "prediction_interval": {
                        "low": q10,
                        "high": q90,
                        "mid_low": q25,
                        "mid_high": q75,
                        "coverage": "80pct_q10_q90",
                    },
                }
            horizons_out[f"{h}d"] = {
                "horizon": h,
                "horizon_label": TFT_HORIZON_LABELS.get(h, f"{h} Trading Days"),
                "targets": block,
                **block,
            }
        return {
            "horizons": horizons_out,
            "attention_weights": attention,
            "version": self.version + "+heuristic",
            "backend": "heuristic",
            "lookback": self.lookback,
            "targets": list(self.targets),
            "target_labels": {t: TFT_TARGET_LABELS.get(t, t) for t in self.targets},
            "horizon_list": list(self.horizons),
            "horizon_labels": {str(h): TFT_HORIZON_LABELS.get(h, f"{h}d") for h in self.horizons},
            "quantiles": list(self.quantiles),
            "outputs": list(TFT_OUTPUTS),
        }

    # ---- backward-compatible stub API ----
    def predict_proba_sequence(self, window: np.ndarray) -> dict[int, float]:
        if window.size == 0:
            return {h: 0.0 for h in self.horizons}
        signal = float(np.clip(np.nanmean(window[-5:]), 0.0, 1.0))
        return {h: float(min(0.95, signal * (0.9 + 0.01 * h))) for h in self.horizons}

    def forecast_paths(self, features: pd.DataFrame) -> dict[str, Any]:
        """Legacy path format + rich TFT block."""
        rich = self.forecast(features)
        # legacy squeeze-affinity style horizons list
        legacy = []
        for h in self.horizons:
            block = (rich.get("horizons") or {}).get(f"{h}d") or {}
            price = block.get("future_price") or {}
            spot = float(features.iloc[-1].get("spot") or 0.0) if not features.empty else 0.0
            med = price.get("median")
            if med is not None and spot > 0:
                expected = (float(med) / spot) - 1.0
                p10 = (float(price.get("q10") or med) / spot) - 1.0
                p90 = (float(price.get("q90") or med) / spot) - 1.0
            else:
                expected = p10 = p90 = 0.0
            legacy.append(
                {
                    "horizon_days": h,
                    "expected_return_pct": expected * 100.0,
                    "p10_pct": p10 * 100.0,
                    "p50_pct": expected * 100.0,
                    "p90_pct": p90 * 100.0,
                    "targets": block,
                }
            )
        rich["paths"] = legacy
        return rich


# Backward-compatible alias
TemporalSqueezeModelStub = TFTForecastModel


def save_tft_model(model: TFTForecastModel, *, root: Path | None = None) -> Path:
    path = (root or resolve_models_root()) / "tft_forecasting" / "latest.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_tft_model(*, root: Path | None = None) -> TFTForecastModel | None:
    path = (root or resolve_models_root()) / "tft_forecasting" / "latest.joblib"
    if not path.is_file():
        return None
    return joblib.load(path)


def train_deep_backend(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    model = TFTForecastModel(feature_columns=feature_columns)
    metrics = model.fit(features, labels)
    if metrics.get("status") == "ok":
        save_tft_model(model)
    return {"versions": {"deep": model.version}, "metrics": metrics, "model": model}


register_backend("deep", train_deep_backend)
