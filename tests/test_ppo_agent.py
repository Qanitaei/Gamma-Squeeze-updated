import numpy as np
import pandas as pd

from gamma_squeeze.rl.actions import N_ACTIONS, ACTION_NAMES
from gamma_squeeze.rl.env import SqueezePortfolioEnv
from gamma_squeeze.rl.ppo import HAS_SB3, PPOAgent, observe_latest
from gamma_squeeze.rl.reward import compute_reward, rolling_sharpe, rolling_sortino
from gamma_squeeze.rl.state import STATE_DIM, STATE_GROUPS, build_observation, state_meta


def _toy_features(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2025-01-02")
    rows = []
    spot = 100.0
    for i in range(n):
        ret = float(rng.normal(0.001, 0.015))
        spot *= 1.0 + ret
        rows.append(
            {
                "symbol": "TEST",
                "as_of": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "spot": spot,
                "ret_1d": ret,
                "ret_5d": ret * 3,
                "rvol_10d": 0.22,
                "net_gex": -2e5,
                "positioning_stress": 0.7,
                "regime_neg_gamma": 1.0,
                "atm_iv": 0.30,
                "dealer_delta": -15000,
                "dealer_gamma": -800,
                "dealer_hedge_requirement": -12000,
                "dealer_liquidity_score": 0.55,
                "dealer_charm": 10.0,
                "dealer_vanna": 5.0,
                "dealer_vomma": 2.0,
                "dealer_theta": -20.0,
                "dealer_vega": 40.0,
                "vix": 22.0,
                "yield_spread": 0.3,
                "fed_funds": 4.25,
                "consumer_sentiment": 60.0,
                "expected_move": 2.5,
            }
        )
    return pd.DataFrame(rows)


def test_state_dim_and_groups():
    from gamma_squeeze.rl.state import STATE_GROUP_LABELS

    meta = state_meta()
    assert meta["state_dim"] == STATE_DIM
    assert list(meta["groups"]) == list(STATE_GROUPS)
    assert meta["group_labels"]["dealer_position"] == "Dealer Position"
    assert STATE_GROUP_LABELS["greeks"] == "Greeks"
    obs = build_observation(_toy_features().iloc[-1])
    assert obs.shape == (STATE_DIM,)
    assert obs.dtype == np.float32


def test_action_catalog():
    assert N_ACTIONS == 13
    assert ACTION_NAMES[0] == "Long"
    assert ACTION_NAMES[2] == "Calls"
    assert ACTION_NAMES[6] == "Iron Condor"
    assert ACTION_NAMES[10] == "Cash"
    assert ACTION_NAMES[11] == "Increase Hedge"
    assert ACTION_NAMES[12] == "Reduce Hedge"
    expected = [
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
    ]
    assert [ACTION_NAMES[i] for i in range(N_ACTIONS)] == expected


def test_reward_components():
    from gamma_squeeze.rl.reward import REWARD_COMPONENT_LABELS, REWARD_COMPONENTS, reward_meta

    rets = [0.01, -0.005, 0.02, -0.01, 0.015]
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    b = compute_reward(
        returns=rets,
        equity_curve=eq,
        transaction_cost=0.0005,
        slippage=0.0003,
    )
    d = b.to_dict()
    for key in REWARD_COMPONENTS:
        assert key in d
    assert rolling_sharpe(rets) == b.sharpe
    assert rolling_sortino(rets) == b.sortino
    assert b.drawdown >= 0
    meta = reward_meta()
    assert meta["component_labels"]["tail_risk"] == "Tail Risk"
    assert REWARD_COMPONENT_LABELS["portfolio_variance"] == "Portfolio Variance"


def test_env_step_and_spaces():
    feat = _toy_features()
    env = SqueezePortfolioEnv(feat, horizon=5, seed=1)
    obs, info = env.reset()
    assert obs.shape == (STATE_DIM,)
    assert env.action_space.n == N_ACTIONS
    total = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, step_info = env.step(0)  # Long
        total += reward
        done = terminated or truncated
        assert "reward_breakdown" in step_info
    assert len(env._returns) == len(feat)


def test_ppo_heuristic_act():
    feat = _toy_features()
    obs = observe_latest(
        feat,
        forecast={"probability": 0.8, "magnitude": 5.0, "expected_duration": 4, "expected_start": 0, "confidence": 0.7},
    )
    agent = PPOAgent()
    action = agent.act(obs)
    assert 0 <= action < N_ACTIONS
    info = agent.act_with_info(obs)
    assert info["action"] in ACTION_NAMES.values()


def test_ppo_sb3_short_train(tmp_path, monkeypatch):
    assert HAS_SB3, "stable-baselines3 required for Phase 11 (pip install -r requirements-rl.txt)"
    from gamma_squeeze.rl import ppo as ppo_mod

    monkeypatch.setattr(ppo_mod, "resolve_models_root", lambda: tmp_path)
    feat = _toy_features(48)
    env = SqueezePortfolioEnv(feat, horizon=3, seed=2)
    agent = PPOAgent()
    result = agent.train(env, timesteps=512, n_steps=128, batch_size=64, seed=2)
    assert result.backend == "stable-baselines3"
    assert result.timesteps >= 128
    assert (tmp_path / "ppo" / "latest.zip").is_file()
    loaded = PPOAgent.load_latest(tmp_path)
    assert loaded.model is not None
    action = loaded.act(observe_latest(feat))
    assert 0 <= action < N_ACTIONS


def test_ppo_api_meta():
    from fastapi.testclient import TestClient

    from services.ppo_agent.app import app

    client = TestClient(app)
    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert body["library"] == "Stable-Baselines3 PPO"
    assert body["sb3_available"] is True
    assert "Dealer Position" in body["state_space"]["group_labels"].values()
    assert len(body["actions"]) == 13
    assert "Sharpe" in body["reward"]["labels"]


def test_observation_sanitizes_nan():
    row = _toy_features().iloc[-1].to_dict()
    row["net_gex"] = float("nan")
    row["atm_iv"] = float("inf")
    obs = build_observation(row)
    assert np.isfinite(obs).all()
