"""Reinforcement learning — Stable-Baselines3 PPO trading agent."""

from gamma_squeeze.rl.actions import ACTION_NAMES, ACTIONS, N_ACTIONS, action_meta
from gamma_squeeze.rl.env import SqueezePortfolioEnv, SqueezeTradeEnv
from gamma_squeeze.rl.ppo import PPOAgent, HAS_SB3
from gamma_squeeze.rl.reward import (
    REWARD_COMPONENT_LABELS,
    REWARD_COMPONENTS,
    RewardWeights,
    compute_reward,
    reward_meta,
)
from gamma_squeeze.rl.state import (
    STATE_DIM,
    STATE_GROUP_LABELS,
    STATE_GROUPS,
    build_observation,
    state_meta,
)

__all__ = [
    "ACTION_NAMES",
    "ACTIONS",
    "HAS_SB3",
    "N_ACTIONS",
    "PPOAgent",
    "REWARD_COMPONENTS",
    "REWARD_COMPONENT_LABELS",
    "RewardWeights",
    "STATE_DIM",
    "STATE_GROUPS",
    "STATE_GROUP_LABELS",
    "SqueezePortfolioEnv",
    "SqueezeTradeEnv",
    "action_meta",
    "build_observation",
    "compute_reward",
    "reward_meta",
    "state_meta",
]
