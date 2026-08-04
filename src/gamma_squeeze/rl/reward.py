"""Composite reward: Sharpe, Sortino, drawdown, costs, slippage, tail risk, variance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass
class RewardWeights:
    sharpe: float = 1.0
    sortino: float = 0.75
    drawdown: float = 1.25
    transaction_cost: float = 0.5
    slippage: float = 0.35
    tail_risk: float = 1.0
    portfolio_variance: float = 0.6


@dataclass
class RewardBreakdown:
    sharpe: float = 0.0
    sortino: float = 0.0
    drawdown: float = 0.0
    transaction_cost: float = 0.0
    slippage: float = 0.0
    tail_risk: float = 0.0
    portfolio_variance: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_std(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    s = float(np.std(x, ddof=1))
    return s if s > 1e-12 else 0.0


def rolling_sharpe(returns: Sequence[float], *, eps: float = 1e-8) -> float:
    r = np.asarray(list(returns), dtype=float)
    if r.size == 0:
        return 0.0
    mu = float(np.mean(r))
    sd = _safe_std(r)
    if sd <= 0:
        return float(np.tanh(mu * 50.0))
    return float(np.tanh((mu / (sd + eps)) / 3.0))


def rolling_sortino(returns: Sequence[float], *, eps: float = 1e-8) -> float:
    r = np.asarray(list(returns), dtype=float)
    if r.size == 0:
        return 0.0
    mu = float(np.mean(r))
    downside = r[r < 0.0]
    dd = _safe_std(downside) if downside.size else 0.0
    if dd <= 0:
        return float(np.tanh(max(mu, 0.0) * 50.0))
    return float(np.tanh((mu / (dd + eps)) / 3.0))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    eq = np.asarray(list(equity_curve), dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-9)
    return float(np.clip(np.max(dd), 0.0, 1.0))


def portfolio_variance(returns: Sequence[float]) -> float:
    r = np.asarray(list(returns), dtype=float)
    if r.size < 2:
        return 0.0
    return float(np.clip(np.var(r, ddof=1) * 250.0, 0.0, 5.0))  # ann. scale soft


def tail_risk(returns: Sequence[float], *, q: float = 0.05) -> float:
    """Positive penalty from left-tail (CVaR-like) magnitude."""
    r = np.asarray(list(returns), dtype=float)
    if r.size == 0:
        return 0.0
    cutoff = float(np.quantile(r, q))
    tail = r[r <= cutoff]
    if tail.size == 0:
        return 0.0
    cvar = float(-np.mean(tail))
    return float(np.clip(cvar * 20.0, 0.0, 1.0))


def compute_reward(
    *,
    returns: Sequence[float],
    equity_curve: Sequence[float],
    transaction_cost: float,
    slippage: float,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    w = weights or RewardWeights()
    sharpe = rolling_sharpe(returns)
    sortino = rolling_sortino(returns)
    dd = max_drawdown(equity_curve)
    var = portfolio_variance(returns)
    tail = tail_risk(returns)
    tc = float(max(0.0, transaction_cost))
    slip = float(max(0.0, slippage))

    total = (
        w.sharpe * sharpe
        + w.sortino * sortino
        - w.drawdown * dd
        - w.transaction_cost * tc
        - w.slippage * slip
        - w.tail_risk * tail
        - w.portfolio_variance * min(var, 1.0)
    )
    # step-local shaping: include last return softly
    if returns:
        total += 0.35 * float(np.tanh(float(returns[-1]) * 40.0))

    return RewardBreakdown(
        sharpe=sharpe,
        sortino=sortino,
        drawdown=dd,
        transaction_cost=tc,
        slippage=slip,
        tail_risk=tail,
        portfolio_variance=var,
        total=float(total),
    )


REWARD_COMPONENTS = (
    "sharpe",
    "sortino",
    "drawdown",
    "transaction_cost",
    "slippage",
    "tail_risk",
    "portfolio_variance",
)

REWARD_COMPONENT_LABELS = {
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "drawdown": "Drawdown",
    "transaction_cost": "Transaction Cost",
    "slippage": "Slippage",
    "tail_risk": "Tail Risk",
    "portfolio_variance": "Portfolio Variance",
}


def reward_meta(weights: RewardWeights | None = None) -> dict[str, object]:
    w = weights or RewardWeights()
    return {
        "components": list(REWARD_COMPONENTS),
        "component_labels": dict(REWARD_COMPONENT_LABELS),
        "labels": [REWARD_COMPONENT_LABELS[c] for c in REWARD_COMPONENTS],
        "weights": asdict(w),
    }
