"""Stable-Baselines3 PPO agent: train, persist, act."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gamma_squeeze.config import resolve_models_root
from gamma_squeeze.rl.actions import ACTION_NAMES, N_ACTIONS, action_meta
from gamma_squeeze.rl.env import SqueezePortfolioEnv
from gamma_squeeze.rl.reward import RewardWeights, reward_meta
from gamma_squeeze.rl.state import STATE_DIM, build_observation, state_meta

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    HAS_SB3 = True
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    HAS_SB3 = False


VERSION = "rl-ppo-sb3-v1"


@dataclass
class PPOTrainResult:
    version: str = VERSION
    backend: str = "stable-baselines3"
    timesteps: int = 0
    n_episodes_eval: int = 0
    mean_reward: float = 0.0
    mean_equity: float = 1.0
    model_path: str | None = None
    state_dim: int = STATE_DIM
    n_actions: int = N_ACTIONS
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ppo_model_dir(root: Path | None = None) -> Path:
    return (root or resolve_models_root()) / "ppo"


def latest_model_path(root: Path | None = None) -> Path:
    return ppo_model_dir(root) / "latest.zip"


def _heuristic_action(obs: np.ndarray) -> int:
    """Fallback policy when no trained model is available."""
    # Layout: dealer(6) regime(5) forecast(5) portfolio(5) risk(4) exposure(4) iv(3) greeks(5) macro(4)
    dealer_stress = float(obs[4]) if obs.size > 4 else 0.0
    neg_gamma = float(obs[6]) if obs.size > 6 else 0.0
    squeeze_p = float(obs[11]) if obs.size > 11 else 0.0
    drawdown = float(obs[21]) if obs.size > 21 else 0.0
    vix_term = float(obs[37]) if obs.size > 37 else 0.0

    if drawdown > 0.12:
        return 11  # Increase Hedge
    if squeeze_p > 0.55 and (dealer_stress > 0.5 or neg_gamma > 0.5):
        return 0  # Long
    if squeeze_p > 0.4 and neg_gamma > 0.4:
        return 2  # Calls
    if vix_term > 0.4 and squeeze_p < 0.35:
        return 6  # Iron Condor (short vol)
    if squeeze_p < 0.25 and dealer_stress < 0.3:
        return 10  # Cash
    return 4  # Debit Spread


class PPOAgent:
    version = VERSION

    def __init__(self):
        self.model = None
        self.backend = "stable-baselines3" if HAS_SB3 else "heuristic"
        self.reward_weights = RewardWeights()

    @classmethod
    def load_latest(cls, root: Path | None = None) -> "PPOAgent":
        agent = cls()
        path = latest_model_path(root)
        if HAS_SB3 and path.is_file():
            agent.model = PPO.load(str(path))
            agent.backend = "stable-baselines3"
        return agent

    def save(self, root: Path | None = None) -> Path | None:
        if self.model is None or not HAS_SB3:
            return None
        d = ppo_model_dir(root)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "latest.zip"
        self.model.save(str(path.with_suffix("")))  # SB3 adds .zip
        meta = {
            "version": self.version,
            "backend": self.backend,
            "state": state_meta(),
            "actions": action_meta(),
            "reward": reward_meta(self.reward_weights),
        }
        with (d / "latest.meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        return path

    def train(
        self,
        env: SqueezePortfolioEnv,
        *,
        timesteps: int = 2048,
        learning_rate: float = 3e-4,
        n_steps: int = 256,
        batch_size: int = 64,
        seed: int = 42,
    ) -> PPOTrainResult:
        if not HAS_SB3:
            return PPOTrainResult(
                backend="heuristic",
                timesteps=0,
                meta={"error": "stable-baselines3 not installed"},
            )

        def _make():
            # Clone-like: reuse same panels
            return SqueezePortfolioEnv(
                env.features,
                env.labels,
                horizon=env.horizon,
                forecast=env.forecast,
                reward_weights=env.reward_weights,
                transaction_cost_mult=env.transaction_cost_mult,
                slippage_bps=env.slippage_bps,
                seed=seed,
            )

        vec = DummyVecEnv([_make])
        # Keep n_steps modest for short panels
        steps = max(64, min(n_steps, max(64, len(env.features))))
        model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=learning_rate,
            n_steps=steps,
            batch_size=min(batch_size, steps),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            seed=seed,
        )
        model.learn(total_timesteps=int(max(steps, timesteps)))
        self.model = model
        self.backend = "stable-baselines3"
        self.reward_weights = env.reward_weights

        # Quick eval
        mean_r, mean_eq, n_ep = self.evaluate(env, n_episodes=1)
        path = self.save()
        return PPOTrainResult(
            timesteps=int(max(steps, timesteps)),
            n_episodes_eval=n_ep,
            mean_reward=mean_r,
            mean_equity=mean_eq,
            model_path=str(path) if path else None,
            meta={"n_steps": steps, "seed": seed},
        )

    def evaluate(self, env: SqueezePortfolioEnv, *, n_episodes: int = 1) -> tuple[float, float, int]:
        rewards: list[float] = []
        equities: list[float] = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            total = 0.0
            while not done:
                action = self.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            rewards.append(total)
            equities.append(float(info.get("equity") or env._equity))
        return float(np.mean(rewards)), float(np.mean(equities)), n_episodes

    def act(self, obs: np.ndarray, *, deterministic: bool = True) -> int:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if x.size != STATE_DIM:
            # pad / truncate
            if x.size < STATE_DIM:
                x = np.pad(x, (0, STATE_DIM - x.size))
            else:
                x = x[:STATE_DIM]
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.model is not None and HAS_SB3:
            action, _ = self.model.predict(x, deterministic=deterministic)
            return int(action)
        return _heuristic_action(x)

    def act_with_info(self, obs: np.ndarray, *, deterministic: bool = True) -> dict[str, Any]:
        action_id = self.act(obs, deterministic=deterministic)
        return {
            "action_id": action_id,
            "action": ACTION_NAMES.get(action_id, "Cash"),
            "version": self.version,
            "backend": self.backend if self.model is not None else ("heuristic" if not HAS_SB3 else "heuristic-untrained"),
            "deterministic": deterministic,
        }


def build_env_from_panels(
    features: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    *,
    horizon: int = 5,
    forecast: dict[str, float] | None = None,
) -> SqueezePortfolioEnv:
    return SqueezePortfolioEnv(
        features=features,
        labels=labels if labels is not None else pd.DataFrame(),
        horizon=horizon,
        forecast=forecast or {},
    )


def observe_latest(
    features: pd.DataFrame,
    *,
    forecast: dict[str, float] | None = None,
    portfolio: dict[str, float] | None = None,
) -> np.ndarray:
    if features.empty:
        return np.zeros(STATE_DIM, dtype=np.float32)
    row = features.sort_values("as_of").iloc[-1]
    return build_observation(row, forecast=forecast, portfolio=portfolio)
