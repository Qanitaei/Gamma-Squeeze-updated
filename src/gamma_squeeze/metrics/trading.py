"""Trading / portfolio performance metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _returns(x: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float).reshape(-1)


def equity_from_returns(returns: np.ndarray, *, start: float = 1.0) -> np.ndarray:
    r = _returns(returns)
    if len(r) == 0:
        return np.asarray([start], dtype=float)
    return start * np.cumprod(1.0 + r)


def sharpe_ratio(returns: Sequence[float] | np.ndarray, *, periods_per_year: float = 252.0) -> float:
    r = _returns(returns)
    if len(r) < 2:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float] | np.ndarray,
    *,
    periods_per_year: float = 252.0,
    target: float = 0.0,
) -> float:
    r = _returns(returns)
    if len(r) < 2:
        return 0.0
    excess = r - target
    downside = excess[excess < 0.0]
    dd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    if dd <= 1e-12:
        return 0.0 if float(np.mean(excess)) <= 0 else float("inf")
    return float((float(np.mean(excess)) / dd) * np.sqrt(periods_per_year))


def maximum_drawdown(equity: Sequence[float] | np.ndarray | None = None, *, returns: Sequence[float] | np.ndarray | None = None) -> float:
    if equity is None:
        if returns is None:
            return 0.0
        eq = equity_from_returns(_returns(returns))
    else:
        eq = np.asarray(list(equity), dtype=float).reshape(-1)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def annual_return(returns: Sequence[float] | np.ndarray, *, periods_per_year: float = 252.0) -> float:
    r = _returns(returns)
    if len(r) == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    years = len(r) / periods_per_year
    if years <= 0:
        return 0.0
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def calmar_ratio(returns: Sequence[float] | np.ndarray, *, periods_per_year: float = 252.0) -> float:
    ann = annual_return(returns, periods_per_year=periods_per_year)
    mdd = maximum_drawdown(returns=returns)
    if mdd <= 1e-12:
        return 0.0 if ann <= 0 else float("inf")
    return float(ann / mdd)


def win_rate(trade_pnls: Sequence[float] | np.ndarray) -> float:
    p = _returns(trade_pnls)
    if len(p) == 0:
        return 0.0
    return float(np.mean(p > 0))


def profit_factor(trade_pnls: Sequence[float] | np.ndarray) -> float:
    p = _returns(trade_pnls)
    gains = float(np.sum(p[p > 0]))
    losses = float(-np.sum(p[p < 0]))
    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def average_trade(trade_pnls: Sequence[float] | np.ndarray) -> float:
    p = _returns(trade_pnls)
    if len(p) == 0:
        return 0.0
    return float(np.mean(p))


def average_hold_time(hold_periods: Sequence[float] | np.ndarray) -> float:
    h = _returns(hold_periods)
    if len(h) == 0:
        return 0.0
    return float(np.mean(h))


def trades_from_returns(
    returns: Sequence[float] | np.ndarray,
    *,
    signal: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build trade PnLs and hold lengths from period returns.
    If signal provided (0/1 or signed), group consecutive active bars into trades.
    Otherwise treat each non-zero return bar as a 1-period trade.
    """
    r = _returns(returns)
    if len(r) == 0:
        return np.zeros(0), np.zeros(0)
    if signal is None:
        mask = np.abs(r) > 0
        return r[mask], np.ones(int(mask.sum()), dtype=float)

    s = np.asarray(signal, dtype=float).reshape(-1)
    n = min(len(r), len(s))
    r, s = r[:n], s[:n]
    active = s != 0
    pnls: list[float] = []
    holds: list[float] = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        segment = r[i:j]
        pnls.append(float(np.prod(1.0 + segment) - 1.0))
        holds.append(float(j - i))
        i = j
    return np.asarray(pnls, dtype=float), np.asarray(holds, dtype=float)


def trading_metrics(
    returns: Sequence[float] | np.ndarray,
    *,
    signal: Sequence[float] | np.ndarray | None = None,
    trade_pnls: Sequence[float] | np.ndarray | None = None,
    hold_periods: Sequence[float] | np.ndarray | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    r = _returns(returns)
    if trade_pnls is None:
        pnls, holds = trades_from_returns(r, signal=signal)
    else:
        pnls = _returns(trade_pnls)
        holds = _returns(hold_periods) if hold_periods is not None else np.ones(len(pnls), dtype=float)

    sharpe = sharpe_ratio(r, periods_per_year=periods_per_year)
    sortino = sortino_ratio(r, periods_per_year=periods_per_year)
    calmar = calmar_ratio(r, periods_per_year=periods_per_year)
    pf = profit_factor(pnls)

    def _finite(x: float) -> float | None:
        if x == float("inf") or x == float("-inf") or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)

    return {
        "status": "ok" if len(r) else "empty",
        "n_periods": int(len(r)),
        "n_trades": int(len(pnls)),
        "Sharpe Ratio": _finite(sharpe),
        "Sortino Ratio": _finite(sortino) if sortino != float("inf") else None,
        "Calmar Ratio": _finite(calmar) if calmar != float("inf") else None,
        "Maximum Drawdown": maximum_drawdown(returns=r),
        "Annual Return": annual_return(r, periods_per_year=periods_per_year),
        "Win Rate": win_rate(pnls),
        "Profit Factor": _finite(pf) if pf != float("inf") else None,
        "Average Trade": average_trade(pnls),
        "Average Hold Time": average_hold_time(holds),
    }
