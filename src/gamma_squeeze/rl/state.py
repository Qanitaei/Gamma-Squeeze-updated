"""State-space builder for PPO observations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STATE_GROUPS: tuple[str, ...] = (
    "dealer_position",
    "regime",
    "forecast",
    "portfolio",
    "risk",
    "exposure",
    "iv",
    "greeks",
    "macro",
)

STATE_GROUP_LABELS: dict[str, str] = {
    "dealer_position": "Dealer Position",
    "regime": "Regime",
    "forecast": "Forecast",
    "portfolio": "Portfolio",
    "risk": "Risk",
    "exposure": "Exposure",
    "iv": "IV",
    "greeks": "Greeks",
    "macro": "Macro",
}

# Fixed widths per group (concatenated observation)
GROUP_DIMS: dict[str, int] = {
    "dealer_position": 6,
    "regime": 5,
    "forecast": 5,
    "portfolio": 5,
    "risk": 4,
    "exposure": 4,
    "iv": 3,
    "greeks": 5,
    "macro": 4,
}

STATE_DIM = sum(GROUP_DIMS.values())  # 41


def _f(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        v = row.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return default


def _tanh_scale(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(np.tanh(x / scale))


def _pad(vec: list[float], n: int) -> list[float]:
    if len(vec) >= n:
        return vec[:n]
    return vec + [0.0] * (n - len(vec))


def build_state_groups(
    row: dict[str, Any] | pd.Series,
    *,
    portfolio: dict[str, float] | None = None,
    forecast: dict[str, float] | None = None,
    risk: dict[str, float] | None = None,
) -> dict[str, list[float]]:
    """Build named state groups from feature row + live portfolio/forecast/risk."""
    r = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    pf = portfolio or {}
    fc = forecast or {}
    rk = risk or {}

    dealer = _pad(
        [
            _tanh_scale(_f(r, "net_gex", "gex_per_spot"), 1e6),
            _tanh_scale(_f(r, "dealer_delta"), 1e5),
            _tanh_scale(_f(r, "dealer_gamma", "gamma_exposure"), 1e4),
            _tanh_scale(_f(r, "dealer_hedge_requirement"), 1e5),
            float(np.clip(_f(r, "dealer_exhaustion", "positioning_stress"), 0, 1)),
            float(np.clip(_f(r, "dealer_liquidity", "dealer_liquidity_score", default=0.5), 0, 1)),
        ],
        GROUP_DIMS["dealer_position"],
    )

    # Regime: squeeze flags + confidence proxy
    regime = _pad(
        [
            float(np.clip(_f(r, "regime_neg_gamma"), 0, 1)),
            float(np.clip(_f(r, "regime_pos_gamma", default=0.0), 0, 1)),
            float(np.clip(_f(r, "positioning_stress"), 0, 1)),
            float(np.clip(_f(r, "hmm_confidence", "regime_confidence", default=0.5), 0, 1)),
            _tanh_scale(_f(r, "rvol_10d", default=0.2) - 0.2, 0.3),
        ],
        GROUP_DIMS["regime"],
    )

    forecast_g = _pad(
        [
            float(np.clip(fc.get("probability", _f(r, "squeeze_probability")), 0, 1)),
            _tanh_scale(fc.get("magnitude", _f(r, "expected_move", default=0.0)), 10.0),
            _tanh_scale(fc.get("expected_duration", _f(r, "expected_duration", default=5.0)), 10.0),
            _tanh_scale(fc.get("expected_start", 0.0), 5.0),
            float(np.clip(fc.get("confidence", _f(r, "forecast_confidence", default=0.5)), 0, 1)),
        ],
        GROUP_DIMS["forecast"],
    )

    portfolio_g = _pad(
        [
            _tanh_scale(pf.get("equity", 1.0) - 1.0, 0.5),
            float(np.clip(pf.get("cash_frac", 1.0), 0, 1)),
            float(np.clip(pf.get("position", 0.0), -1, 1)),
            float(np.clip(pf.get("hedge_frac", 0.0), 0, 1)),
            _tanh_scale(pf.get("pnl", 0.0), 0.2),
        ],
        GROUP_DIMS["portfolio"],
    )

    risk_g = _pad(
        [
            float(np.clip(rk.get("drawdown", 0.0), 0, 1)),
            _tanh_scale(rk.get("realized_vol", _f(r, "rvol_10d", default=0.2)), 0.5),
            _tanh_scale(rk.get("variance", 0.0), 0.05),
            float(np.clip(rk.get("tail_risk", 0.0), 0, 1)),
        ],
        GROUP_DIMS["risk"],
    )

    exposure = _pad(
        [
            _tanh_scale(pf.get("delta_exposure", pf.get("position", 0.0)), 1.0),
            _tanh_scale(pf.get("gamma_exposure", 0.0), 1.0),
            _tanh_scale(pf.get("vega_exposure", 0.0), 1.0),
            _tanh_scale(_f(r, "net_gex"), 1e6),
        ],
        GROUP_DIMS["exposure"],
    )

    iv = _pad(
        [
            _tanh_scale(_f(r, "atm_iv", "iv", default=0.25) - 0.25, 0.25),
            _tanh_scale(_f(r, "iv_rank", "iv_percentile", default=0.5) - 0.5, 0.5),
            _tanh_scale(_f(r, "rvol_10d", default=0.2) - _f(r, "atm_iv", default=0.25), 0.2),
        ],
        GROUP_DIMS["iv"],
    )

    greeks = _pad(
        [
            _tanh_scale(_f(r, "dealer_charm"), 1e4),
            _tanh_scale(_f(r, "dealer_vanna"), 1e4),
            _tanh_scale(_f(r, "dealer_vomma"), 1e4),
            _tanh_scale(_f(r, "dealer_theta"), 1e4),
            _tanh_scale(_f(r, "dealer_vega"), 1e4),
        ],
        GROUP_DIMS["greeks"],
    )

    macro = _pad(
        [
            _tanh_scale(_f(r, "vix", default=18.0) - 18.0, 15.0),
            _tanh_scale(_f(r, "yield_spread", default=0.0), 2.0),
            _tanh_scale(_f(r, "fed_funds", default=4.0) - 4.0, 3.0),
            _tanh_scale(_f(r, "consumer_sentiment", default=70.0) - 70.0, 30.0),
        ],
        GROUP_DIMS["macro"],
    )

    return {
        "dealer_position": dealer,
        "regime": regime,
        "forecast": forecast_g,
        "portfolio": portfolio_g,
        "risk": risk_g,
        "exposure": exposure,
        "iv": iv,
        "greeks": greeks,
        "macro": macro,
    }


def flatten_state(groups: dict[str, list[float]]) -> np.ndarray:
    parts: list[float] = []
    for name in STATE_GROUPS:
        parts.extend(_pad(list(groups.get(name) or []), GROUP_DIMS[name]))
    arr = np.asarray(parts, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)


def build_observation(
    row: dict[str, Any] | pd.Series,
    *,
    portfolio: dict[str, float] | None = None,
    forecast: dict[str, float] | None = None,
    risk: dict[str, float] | None = None,
) -> np.ndarray:
    return flatten_state(
        build_state_groups(row, portfolio=portfolio, forecast=forecast, risk=risk)
    )


def state_meta() -> dict[str, Any]:
    return {
        "groups": list(STATE_GROUPS),
        "group_labels": dict(STATE_GROUP_LABELS),
        "group_dims": dict(GROUP_DIMS),
        "state_dim": STATE_DIM,
    }
