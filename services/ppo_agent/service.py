from __future__ import annotations

from typing import Any

import pandas as pd

from gamma_squeeze.features.feature_store import build_features_for_symbol, load_features
from gamma_squeeze.labels.label_builder import build_labels_for_symbol, load_labels
from gamma_squeeze.rl.actions import ACTION_NAMES, action_meta
from gamma_squeeze.rl.ppo import (
    HAS_SB3,
    PPOAgent,
    build_env_from_panels,
    observe_latest,
)
from gamma_squeeze.rl.reward import reward_meta
from gamma_squeeze.rl.state import state_meta


def _forecast_block(feat: pd.DataFrame, composite: dict[str, float] | None = None) -> dict[str, float]:
    if composite:
        return composite
    row = feat.sort_values("as_of").iloc[-1]
    stress = float(row.get("positioning_stress") or 0.0)
    neg = float(row.get("regime_neg_gamma") or 0.0)
    return {
        "probability": max(0.0, min(1.0, 0.55 * stress + 0.45 * neg)),
        "magnitude": float(row.get("expected_move") or abs(row.get("ret_5d") or 0.0) * 100.0 or 0.0),
        "expected_duration": 5.0,
        "expected_start": 0.0 if stress > 0.65 else 2.0,
        "confidence": 0.45,
    }


def _maybe_composite_forecast(symbol: str, feat: pd.DataFrame) -> dict[str, float] | None:
    try:
        from gamma_squeeze.models.composite_squeeze import run_composite_squeeze

        result = run_composite_squeeze(symbol, feat, matrix=None)
        return {
            "probability": float(result.probability),
            "magnitude": float(result.magnitude),
            "expected_duration": float(result.expected_duration),
            "expected_start": float(result.expected_start),
            "confidence": float(result.confidence),
        }
    except Exception:  # noqa: BLE001
        return None


def run_ppo(
    symbol: str,
    *,
    train: bool = False,
    timesteps: int = 2048,
    horizon: int = 5,
    deterministic: bool = True,
    use_composite_forecast: bool = False,
) -> dict[str, Any]:
    sym = symbol.upper()
    feat = load_features(sym)
    if feat.empty:
        feat = build_features_for_symbol(
            sym,
            include_skew=False,
            include_macro=True,
            include_technicals=True,
            include_dealer=True,
            include_options_metrics=True,
        )
    if feat.empty:
        return {
            "symbol": sym,
            "error": "no_features",
            "action": "Cash",
            "action_id": 10,
            "backend": "none",
        }

    labs = load_labels(sym)
    if labs.empty:
        try:
            labs = build_labels_for_symbol(sym, features=feat)
        except Exception:  # noqa: BLE001
            pass

    composite = _maybe_composite_forecast(sym, feat) if use_composite_forecast else None
    forecast = _forecast_block(feat, composite)
    env = build_env_from_panels(feat, labs, horizon=horizon, forecast=forecast)
    agent = PPOAgent.load_latest()

    train_metrics: dict[str, Any] | None = None
    if train:
        if HAS_SB3:
            try:
                train_metrics = agent.train(env, timesteps=int(timesteps)).to_dict()
            except Exception as exc:  # noqa: BLE001
                train_metrics = {
                    "backend": "heuristic",
                    "timesteps": 0,
                    "error": f"sb3_train_failed: {exc}",
                }
                agent.model = None
                agent.backend = "heuristic"
        else:
            train_metrics = {"backend": "heuristic", "timesteps": 0, "error": "sb3_missing"}

    obs = observe_latest(feat, forecast=forecast)
    decision = agent.act_with_info(obs, deterministic=deterministic)

    env.reset()
    env._i = max(0, len(feat) - 1)
    _, reward, _, _, info = env.step(decision["action_id"])

    return {
        "symbol": sym,
        "as_of": str(feat.sort_values("as_of").iloc[-1]["as_of"])[:10],
        **decision,
        "forecast": forecast,
        "state": state_meta(),
        "reward_preview": {
            "total": float(reward),
            "breakdown": (info or {}).get("reward_breakdown"),
            "equity": (info or {}).get("equity"),
        },
        "train_metrics": train_metrics,
        "sb3_available": HAS_SB3,
        "actions": ACTION_NAMES,
    }


def ppo_meta() -> dict[str, Any]:
    return {
        "service": "ppo_agent",
        "backend": "stable-baselines3" if HAS_SB3 else "heuristic",
        "sb3_available": HAS_SB3,
        "library": "Stable-Baselines3 PPO",
        "state_space": state_meta(),
        "actions": action_meta(),
        "reward": reward_meta(),
        "action_names": [
            "Long",
            "Short",
            "Calls",
            "Puts",
            "Debit Spread",
            "Credit Spread",
            "Iron Condor",
            "Calendar",
            "Butterfly",
            "Covered Call",
            "Cash",
            "Increase Hedge",
            "Reduce Hedge",
        ],
    }
