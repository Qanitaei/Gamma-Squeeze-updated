"""PPO agent public API + retrain registry hook."""

from __future__ import annotations

from typing import Any

import pandas as pd

from gamma_squeeze.retrain.registry import register_backend
from gamma_squeeze.rl.env import SqueezePortfolioEnv, SqueezeTradeEnv
from gamma_squeeze.rl.ppo import PPOAgent, build_env_from_panels


# Back-compat exports
PPOAgentStub = PPOAgent


def train_rl_backend(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    del feature_columns  # observation built from structured state groups
    env = build_env_from_panels(features, labels, horizon=5)
    agent = PPOAgent()
    # Short train for registry/retrain pipeline
    n = max(256, min(2048, max(128, len(features) * 8)))
    result = agent.train(env, timesteps=n)
    return {
        "versions": {"rl": agent.version},
        "metrics": result.to_dict(),
        "agent": agent,
    }


register_backend("rl", train_rl_backend)

__all__ = [
    "PPOAgent",
    "PPOAgentStub",
    "SqueezePortfolioEnv",
    "SqueezeTradeEnv",
    "train_rl_backend",
]
