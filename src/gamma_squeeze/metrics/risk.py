"""Tail risk metrics — VaR-style tail and Expected Shortfall (CVaR)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _arr(x: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float).reshape(-1)


def tail_risk(
    returns: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> float:
    """
    Left-tail risk: absolute Value-at-Risk at level alpha (positive = loss magnitude).
    """
    r = _arr(returns)
    if len(r) == 0:
        return 0.0
    q = float(np.quantile(r, alpha))
    return float(max(-q, 0.0))


def expected_shortfall(
    returns: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> float:
    """
    Expected Shortfall / CVaR: mean loss beyond VaR (positive = loss magnitude).
    """
    r = _arr(returns)
    if len(r) == 0:
        return 0.0
    q = float(np.quantile(r, alpha))
    tail = r[r <= q]
    if len(tail) == 0:
        return float(max(-q, 0.0))
    return float(max(-float(np.mean(tail)), 0.0))


def risk_metrics(
    returns: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r = _arr(returns)
    return {
        "status": "ok" if len(r) else "empty",
        "n": int(len(r)),
        "alpha": alpha,
        "Tail Risk": tail_risk(r, alpha=alpha),
        "Expected Shortfall": expected_shortfall(r, alpha=alpha),
    }
