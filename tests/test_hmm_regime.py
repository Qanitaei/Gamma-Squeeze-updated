"""Phase 5 — Hidden Markov Model regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from gamma_squeeze.regime.hmm_regime import (
    HMM_OUTPUT_FIELDS,
    N_STATES,
    REGIME_STATES,
    fit_hmm_regimes,
    infer_regime,
)


EXPECTED_STATES = [
    "Bull",
    "Strong Bull",
    "Neutral",
    "Compression",
    "Distribution",
    "Accumulation",
    "Positive Gamma",
    "Negative Gamma",
    "Dealer Neutral",
    "High Volatility",
    "Low Volatility",
    "Gamma Expansion",
    "Gamma Squeeze",
    "Short Squeeze",
    "Capitulation",
    "Panic",
]


def _synth_features(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2025-01-02")
    rows = []
    for i in range(n):
        if i >= n - 10:
            ret, gex, stress, rvol, neg = 0.02 + 0.01 * rng.normal(), -8e5, 0.85, 0.35, 1.0
        elif i < 20:
            ret, gex, stress, rvol, neg = 0.004 + 0.002 * rng.normal(), 4e5, 0.2, 0.12, 0.0
        else:
            ret = 0.001 * rng.normal()
            gex = 5e4 * rng.normal()
            stress, rvol = 0.35, 0.18
            neg = 1.0 if gex < 0 else 0.0
        rows.append(
            {
                "symbol": "TEST",
                "as_of": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "ret_1d": ret,
                "ret_5d": ret * 3,
                "rvol_10d": rvol,
                "positioning_stress": stress,
                "gex_per_spot": gex,
                "net_gex": gex,
                "regime_neg_gamma": neg,
                "pcr_oi": 1.0 + 0.1 * rng.normal(),
                "bollinger_width": 0.05 + 0.02 * (rvol / 0.2),
                "rsi": 50 + 20 * np.tanh(ret * 40),
                "adx": 20 + 10 * abs(ret) * 50,
                "vix": 15 + 40 * max(rvol - 0.1, 0),
            }
        )
    return pd.DataFrame(rows)


def test_hidden_states_match_spec():
    assert REGIME_STATES == EXPECTED_STATES
    assert N_STATES == 16
    assert HMM_OUTPUT_FIELDS == (
        "current_state",
        "transition_matrix",
        "next_state_probability",
        "confidence",
    )


def test_hmm_locked_outputs():
    feat = _synth_features(80)
    model = fit_hmm_regimes(feat)
    out = infer_regime(feat, model)

    for key in HMM_OUTPUT_FIELDS:
        assert key in out

    assert out["current_state"] in REGIME_STATES
    assert 0.0 <= out["confidence"] <= 1.0
    assert set(out["next_state_probability"]) == set(REGIME_STATES)
    assert abs(sum(out["next_state_probability"].values()) - 1.0) < 1e-6
    assert set(out["transition_matrix"]) == set(REGIME_STATES)
    for row in out["transition_matrix"].values():
        assert abs(sum(row.values()) - 1.0) < 1e-6
    assert model.transition_matrix.shape == (16, 16)


def test_squeeze_window_prefers_stress_regimes():
    feat = _synth_features(80)
    out = infer_regime(feat, fit_hmm_regimes(feat))
    # late window is neg-gamma + upside — should not be Low Volatility / Compression
    assert out["current_state"] not in {"Low Volatility", "Compression", "Neutral"}
    top_next = max(out["next_state_probability"], key=out["next_state_probability"].get)
    assert top_next in REGIME_STATES


def test_regime_api_contract():
    from services.regime_hmm.app import app

    client = TestClient(app)
    states = client.get("/v1/states")
    assert states.status_code == 200
    body = states.json()
    assert body["states"] == EXPECTED_STATES
    assert body["outputs"] == list(HMM_OUTPUT_FIELDS)

    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["n_states"] == 16
