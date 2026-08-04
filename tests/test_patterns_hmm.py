import numpy as np
import pandas as pd

from gamma_squeeze.patterns.recognition import detect_patterns
from gamma_squeeze.regime.hmm_regime import (
    N_STATES,
    REGIME_STATES,
    fit_hmm_regimes,
    infer_regime,
)


def test_detect_patterns_breakout():
    n = 50
    close = np.linspace(100, 110, n)
    close[-1] = 120
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1,
            "low": close - 1,
            "volume": np.full(n, 1e6),
        }
    )
    out = detect_patterns(df)
    names = {p["name"] for p in out["patterns"]}
    assert "range_breakout" in names


def _synth_features(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = pd.Timestamp("2025-01-02")
    rows = []
    for i in range(n):
        # inject a late negative-gamma + upside thrust window (squeeze-ish)
        if i >= n - 10:
            ret = 0.02 + 0.01 * rng.normal()
            gex = -8e5
            stress = 0.85
            rvol = 0.35
            neg = 1.0
        elif i < 20:
            ret = 0.004 + 0.002 * rng.normal()
            gex = 4e5
            stress = 0.2
            rvol = 0.12
            neg = 0.0
        else:
            ret = 0.001 * rng.normal()
            gex = 5e4 * rng.normal()
            stress = 0.35
            rvol = 0.18
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


def test_hmm_fit_infer_named_states():
    feat = _synth_features(80)
    model = fit_hmm_regimes(feat)
    out = infer_regime(feat, model)

    assert out["n_obs"] == 80
    assert out["current_state"] in REGIME_STATES
    assert out["regime"] == out["current_state"]
    assert 0.0 <= out["confidence"] <= 1.0
    assert set(out["next_state_probability"]) == set(REGIME_STATES)
    assert abs(sum(out["next_state_probability"].values()) - 1.0) < 1e-6
    assert set(out["transition_matrix"]) == set(REGIME_STATES)
    # rows of transition matrix sum to ~1
    for row in out["transition_matrix"].values():
        assert abs(sum(row.values()) - 1.0) < 1e-6
    assert len(out["state_labels"]) == N_STATES
    assert len(out["path"]) > 0


def test_hmm_transition_square():
    feat = _synth_features(40)
    model = fit_hmm_regimes(feat)
    assert model.transition_matrix is not None
    assert model.transition_matrix.shape == (N_STATES, N_STATES)
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0, atol=1e-6)


def test_hmm_squeeze_window_prefers_risk_on_gamma():
    feat = _synth_features(60)
    out = infer_regime(feat, fit_hmm_regimes(feat))
    # last window is engineered as squeeze-like; accept related gamma/trend states
    assert out["current_state"] in {
        "Gamma Squeeze",
        "Short Squeeze",
        "Negative Gamma",
        "Gamma Expansion",
        "Strong Bull",
        "High Volatility",
        "Bull",
    }
