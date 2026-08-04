import numpy as np
import pandas as pd

from gamma_squeeze.models.composite_squeeze import (
    COMPOSITE_INPUT_LABELS,
    COMPOSITE_INPUTS,
    COMPOSITE_OUTPUT_LABELS,
    COMPOSITE_OUTPUTS,
    COMPOSITE_SCORE_LABELS,
    COMPOSITE_SCORES,
    _compute_scores,
    run_composite_squeeze,
)


def _toy_features(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    base = pd.Timestamp("2025-01-02")
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbol": "TEST",
                "as_of": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "spot": 100 + i * 0.1,
                "ret_1d": float(rng.normal(0.002, 0.01)),
                "ret_5d": float(rng.normal(0.01, 0.02)),
                "rvol_10d": 0.25,
                "net_gex": -5e5,
                "gex_per_spot": -5e5,
                "positioning_stress": 0.8,
                "regime_neg_gamma": 1.0,
                "pcr_oi": 1.2,
                "dealer_liquidity_score": 0.55,
                "dealer_hedge_requirement": -20000,
                "gamma_flip": 98.0,
                "call_wall": 110.0,
                "put_wall": 95.0,
                "atm_iv": 0.32,
                "expected_move": 3.0,
            }
        )
    return pd.DataFrame(rows)


def _toy_matrix() -> dict:
    return {
        "symbol": "TEST",
        "underlying_price": 105.0,
        "as_of_date": "2025-03-01",
        "contracts": [
            {
                "strike_price": 100,
                "put_call": "put",
                "open_interest": 800,
                "gamma": 0.02,
                "delta": -0.4,
                "volatility": 0.3,
                "days_to_expiration": 10,
                "total_volume": 200,
            },
            {
                "strike_price": 105,
                "put_call": "call",
                "open_interest": 1000,
                "gamma": 0.025,
                "delta": 0.5,
                "volatility": 0.28,
                "days_to_expiration": 10,
                "total_volume": 300,
            },
        ],
    }


def test_compute_scores_bounds():
    scores = _compute_scores(
        {
            "feature_row": {
                "positioning_stress": 0.9,
                "regime_neg_gamma": 1.0,
                "net_gex": -1e6,
                "ret_5d": 0.03,
            },
            "hmm": {"current_state": "Gamma Squeeze", "confidence": 0.8},
            "xgboost": {"probability": {"up": 0.7, "flat": 0.2, "down": 0.1}},
            "dealer": {
                "dealer_exhaustion": 0.6,
                "dealer_liquidity": 0.5,
                "share_purchases": 1e5,
                "share_sales": 8e4,
                "gamma_ramp": 5e4,
            },
            "patterns": {
                "top": [{"pattern": "Bull Flag", "confidence": 0.7}],
                "compression": True,
                "n_patterns": 2,
            },
            "macro": {"vix": 22},
            "sentiment": {"squeeze_sentiment": 0.7},
        }
    )
    d = scores.to_dict()
    assert set(d) == set(COMPOSITE_SCORES)
    for v in d.values():
        assert 0.0 <= v <= 1.0


def test_run_composite_squeeze_outputs(monkeypatch):
    # Avoid live OHLCV / macro network in unit test
    import gamma_squeeze.models.composite_squeeze as cs

    def fake_ohlcv(symbol):
        n = 80
        close = np.linspace(100, 110, n)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        )

    monkeypatch.setattr(cs, "fetch_daily_ohlcv", fake_ohlcv)
    monkeypatch.setattr(
        cs,
        "compute_macro_features",
        lambda **kwargs: type(
            "M",
            (),
            {
                "flat_features": lambda self: {
                    "consumer_sentiment": 55.0,
                    "vix": 20.0,
                    "fed_funds": 4.0,
                    "yield_spread": 0.4,
                }
            },
        )(),
    )

    feat = _toy_features()
    result = run_composite_squeeze("TEST", feat, _toy_matrix())
    d = result.to_dict()
    for key in COMPOSITE_OUTPUTS:
        assert key in d
    assert "scores" in d
    assert 0 <= d["probability"] <= 1
    assert 0 <= d["confidence"] <= 1
    assert d["risk_rating"] in {"Low", "Medium", "High", "Extreme"}
    assert set(d["scores"]) == set(COMPOSITE_SCORES)
    assert d["final_output"]["probability"] == d["probability"]
    for key in COMPOSITE_INPUTS:
        assert key in d["inputs"]
    assert d["input_labels"]["hmm"] == "Hidden Markov"
    assert d["score_labels"]["breakout_score"] == "Breakout Score"


def test_locked_constants():
    assert COMPOSITE_INPUT_LABELS["tft"] == "Temporal Transformer"
    assert COMPOSITE_INPUT_LABELS["dealer"] == "Dealer Simulation"
    assert COMPOSITE_SCORE_LABELS["gamma_squeeze_score"] == "Gamma Squeeze Score"
    assert COMPOSITE_OUTPUT_LABELS["expected_duration"] == "Expected Duration"
    assert COMPOSITE_OUTPUTS[-1] == "risk_rating"


def test_squeeze_api_meta():
    from fastapi.testclient import TestClient

    from services.gamma_squeeze_engine.app import app

    client = TestClient(app)
    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert body["inputs"] == list(COMPOSITE_INPUTS)
    assert body["scores"] == list(COMPOSITE_SCORES)
    assert body["outputs"] == list(COMPOSITE_OUTPUTS)
    assert body["input_labels"]["sentiment"] == "Sentiment"
