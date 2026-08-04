"""Gymnasium trading environment for gamma-squeeze PPO."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore

from gamma_squeeze.rl.actions import N_ACTIONS
from gamma_squeeze.rl.reward import RewardBreakdown, RewardWeights, compute_reward
from gamma_squeeze.rl.state import STATE_DIM, build_observation


# Approximate transaction cost (bps of notional) and slippage by action family
_ACTION_COST_BPS = {
    0: 5.0,  # Long
    1: 5.0,  # Short
    2: 15.0,  # Calls
    3: 15.0,  # Puts
    4: 12.0,  # Debit Spread
    5: 12.0,  # Credit Spread
    6: 18.0,  # Iron Condor
    7: 14.0,  # Calendar
    8: 16.0,  # Butterfly
    9: 10.0,  # Covered Call
    10: 1.0,  # Cash
    11: 4.0,  # Increase Hedge
    12: 4.0,  # Reduce Hedge
}

_ACTION_BETA = {
    # signed market beta of the structure (approx)
    0: 1.0,
    1: -1.0,
    2: 0.7,
    3: -0.7,
    4: 0.45,
    5: -0.25,
    6: 0.0,
    7: 0.15,
    8: 0.05,
    9: 0.55,
    10: 0.0,
    11: 0.0,
    12: 0.0,
}


class SqueezePortfolioEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Discrete-action portfolio env over feature/label panels.

    State: dealer, regime, forecast, portfolio, risk, exposure, IV, greeks, macro.
    Actions: Long/Short/Calls/Puts/spreads/condor/calendar/butterfly/covered/cash/hedge.
    Reward: Sharpe − Sortino blend with drawdown, costs, slippage, tail, variance.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame | None = None,
        *,
        horizon: int = 5,
        forecast: dict[str, float] | None = None,
        reward_weights: RewardWeights | None = None,
        transaction_cost_mult: float = 1.0,
        slippage_bps: float = 3.0,
        seed: int | None = None,
    ):
        if gym is None:
            raise ImportError("gymnasium is required: pip install -r requirements-rl.txt")

        self.features = features.reset_index(drop=True).copy()
        self.labels = labels.reset_index(drop=True).copy() if labels is not None else pd.DataFrame()
        self.horizon = int(horizon)
        self.forecast = forecast or {}
        self.reward_weights = reward_weights or RewardWeights()
        self.transaction_cost_mult = float(transaction_cost_mult)
        self.slippage_bps = float(slippage_bps)

        self.observation_space = spaces.Box(
            low=-np.ones(STATE_DIM, dtype=np.float32),
            high=np.ones(STATE_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._i = 0
        self._equity = 1.0
        self._peak = 1.0
        self._cash_frac = 1.0
        self._position = 0.0
        self._hedge_frac = 0.0
        self._delta_exp = 0.0
        self._gamma_exp = 0.0
        self._vega_exp = 0.0
        self._returns: list[float] = []
        self._equity_curve: list[float] = [1.0]
        self._last_action = 10  # Cash
        self._rng = np.random.default_rng(seed)

        # Precompute forward returns (sanitize non-finite labels)
        self._fwd = np.nan_to_num(self._build_forward_returns(), nan=0.0, posinf=0.0, neginf=0.0)

    def _build_forward_returns(self) -> np.ndarray:
        n = len(self.features)
        out = np.zeros(n, dtype=float)
        if self.features.empty:
            return out
        filled = np.zeros(n, dtype=bool)
        # Prefer label column
        col = f"fwd_ret_{self.horizon}d"
        if not self.labels.empty and "as_of" in self.labels.columns and col in self.labels.columns:
            lab = self.labels.copy()
            lab["_as_of"] = lab["as_of"].astype(str).str[:10]
            lab = lab.drop_duplicates("_as_of", keep="last").set_index("_as_of")[col]
            for i, row in self.features.iterrows():
                key = str(row.get("as_of"))[:10]
                if key in lab.index:
                    try:
                        out[int(i)] = float(lab.loc[key])
                        filled[int(i)] = True
                    except Exception:  # noqa: BLE001
                        pass
        # Fallback: rolling sum of ret_1d / spot for unfilled rows
        if not filled.all() and "ret_1d" in self.features.columns:
            r1 = pd.to_numeric(self.features["ret_1d"], errors="coerce").fillna(0.0).to_numpy()
            for i in range(n):
                if not filled[i]:
                    out[i] = float(np.sum(r1[i + 1 : i + 1 + self.horizon]))
        elif not filled.all() and "spot" in self.features.columns:
            spot = pd.to_numeric(self.features["spot"], errors="coerce").to_numpy()
            for i in range(n - self.horizon):
                if not filled[i] and spot[i] > 0 and spot[i + self.horizon] > 0:
                    out[i] = float(spot[i + self.horizon] / spot[i] - 1.0)
        return out

    def _portfolio_dict(self) -> dict[str, float]:
        return {
            "equity": self._equity,
            "cash_frac": self._cash_frac,
            "position": self._position,
            "hedge_frac": self._hedge_frac,
            "pnl": self._equity - 1.0,
            "delta_exposure": self._delta_exp,
            "gamma_exposure": self._gamma_exp,
            "vega_exposure": self._vega_exp,
        }

    def _risk_dict(self) -> dict[str, float]:
        dd = 0.0 if self._peak <= 0 else max(0.0, (self._peak - self._equity) / self._peak)
        var = float(np.var(self._returns[-20:], ddof=1)) if len(self._returns) >= 2 else 0.0
        tail = 0.0
        if self._returns:
            q = float(np.quantile(self._returns, 0.05))
            left = [x for x in self._returns if x <= q]
            if left:
                tail = float(np.clip(-np.mean(left) * 20.0, 0, 1))
        rvol = float(np.std(self._returns[-20:])) if len(self._returns) >= 2 else 0.2
        return {
            "drawdown": dd,
            "realized_vol": rvol,
            "variance": var,
            "tail_risk": tail,
        }

    def _obs(self) -> np.ndarray:
        if self.features.empty:
            return np.zeros(STATE_DIM, dtype=np.float32)
        i = min(self._i, len(self.features) - 1)
        row = self.features.iloc[i]
        obs = build_observation(
            row,
            portfolio=self._portfolio_dict(),
            forecast=self.forecast,
            risk=self._risk_dict(),
        )
        return np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)

    def _apply_action(self, action: int) -> tuple[float, float]:
        """Update portfolio controls; return (transaction_cost, slippage) as fractions."""
        a = int(action)
        prev = self._last_action
        cost_bps = _ACTION_COST_BPS.get(a, 10.0) * self.transaction_cost_mult
        # turnover proxy when action changes
        turnover = 0.0 if a == prev else 0.35 + 0.15 * abs(_ACTION_BETA.get(a, 0) - _ACTION_BETA.get(prev, 0))
        tc = (cost_bps / 1e4) * max(turnover, 0.05)
        slip = (self.slippage_bps / 1e4) * (0.5 + abs(_ACTION_BETA.get(a, 0)))
        # noise
        slip *= float(1.0 + 0.1 * self._rng.normal())

        beta = _ACTION_BETA.get(a, 0.0)
        if a == 10:  # Cash
            self._position = 0.0
            self._cash_frac = 1.0
            self._delta_exp = 0.0
            self._gamma_exp = 0.0
            self._vega_exp = 0.0
        elif a == 11:  # Increase Hedge
            self._hedge_frac = float(min(1.0, self._hedge_frac + 0.15))
            self._delta_exp = self._position * (1.0 - 0.7 * self._hedge_frac)
        elif a == 12:  # Reduce Hedge
            self._hedge_frac = float(max(0.0, self._hedge_frac - 0.15))
            self._delta_exp = self._position * (1.0 - 0.7 * self._hedge_frac)
        else:
            self._position = float(np.clip(beta, -1.0, 1.0))
            self._cash_frac = float(max(0.0, 1.0 - abs(self._position) * 0.8))
            self._delta_exp = self._position * (1.0 - 0.7 * self._hedge_frac)
            # options structures add gamma/vega
            if a in (2, 3, 4, 5, 6, 7, 8, 9):
                self._gamma_exp = float(0.3 if a not in (5, 6) else -0.3)
                self._vega_exp = float(0.4 if a in (2, 3, 4, 7) else -0.35 if a in (5, 6) else 0.1)
            else:
                self._gamma_exp = 0.0
                self._vega_exp = 0.0

        self._last_action = a
        return float(max(0.0, tc)), float(max(0.0, slip))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        del options
        self._i = 0
        self._equity = 1.0
        self._peak = 1.0
        self._cash_frac = 1.0
        self._position = 0.0
        self._hedge_frac = 0.0
        self._delta_exp = 0.0
        self._gamma_exp = 0.0
        self._vega_exp = 0.0
        self._returns = []
        self._equity_curve = [1.0]
        self._last_action = 10
        return self._obs(), {"as_of": self._as_of()}

    def _as_of(self) -> str:
        if self.features.empty:
            return ""
        i = min(self._i, len(self.features) - 1)
        return str(self.features.iloc[i].get("as_of"))[:10]

    def step(self, action: int):
        tc, slip = self._apply_action(int(action))
        i = min(self._i, len(self.features) - 1)
        market_ret = float(self._fwd[i]) if len(self._fwd) else 0.0
        # hedged effective exposure
        exposure = self._delta_exp
        # short-vol structures earn small carry, pay left-tail on large moves
        carry = 0.0
        a = int(action)
        if a in (5, 6, 9):
            carry = 0.0008
            if abs(market_ret) > 0.03:
                carry -= 0.5 * abs(market_ret)
        if a in (2, 3, 4) and abs(market_ret) > 0.02:
            carry += 0.15 * abs(market_ret) * np.sign(exposure if exposure != 0 else _ACTION_BETA.get(a, 0))

        if not np.isfinite(market_ret):
            market_ret = 0.0
        pnl = exposure * market_ret + carry - tc - slip
        if not np.isfinite(pnl):
            pnl = -tc - slip
        self._equity = float(max(1e-6, self._equity * (1.0 + pnl)))
        self._peak = max(self._peak, self._equity)
        self._returns.append(float(pnl))
        self._equity_curve.append(self._equity)

        breakdown: RewardBreakdown = compute_reward(
            returns=self._returns,
            equity_curve=self._equity_curve,
            transaction_cost=tc,
            slippage=slip,
            weights=self.reward_weights,
        )

        self._i += 1
        terminated = self._i >= len(self.features)
        truncated = False
        info = {
            "as_of": self._as_of(),
            "reward_breakdown": breakdown.to_dict(),
            "equity": self._equity,
            "market_ret": market_ret,
            "pnl": pnl,
            "action": int(action),
        }
        return self._obs(), float(breakdown.total), terminated, truncated, info


# Backward-compatible alias used by older registry code
SqueezeTradeEnv = SqueezePortfolioEnv
