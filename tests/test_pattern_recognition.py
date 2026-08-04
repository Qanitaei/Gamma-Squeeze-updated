import numpy as np
import pandas as pd

from gamma_squeeze.patterns.hybrid_encoder import (
    cnn_features,
    hybrid_embedding,
    lstm_features,
    transformer_features,
)
from gamma_squeeze.patterns.recognition import (
    CANDLESTICK_PATTERNS,
    CHART_PATTERNS,
    PATTERN_BACKEND,
    PATTERN_OUTPUTS,
    detect_patterns,
)


def _ohlcv_from_close(close: np.ndarray, noise: float = 0.4) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    high = close + noise
    low = close - noise
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.full(len(close), 1e6)}
    )


def test_breakout_legacy_and_schema():
    n = 50
    close = np.linspace(100, 110, n)
    close[-1] = 120
    out = detect_patterns(_ohlcv_from_close(close), lookback=50, min_confidence=0.3)
    names = {p["pattern"] for p in out["patterns"]}
    assert "range_breakout" in names
    assert out["outputs"] == list(PATTERN_OUTPUTS)
    for p in out["patterns"]:
        assert "pattern" in p
        assert "confidence" in p
        assert "probability" in p
        assert "target_projection" in p
        assert set(p["target_projection"]) >= {"price", "pct", "direction"}


def test_nr7_and_inside_bar():
    n = 30
    close = np.linspace(100, 105, n)
    df = _ohlcv_from_close(close, noise=1.0)
    # make last bar inside + NR7
    df.loc[df.index[-1], "high"] = df.loc[df.index[-2], "high"] - 0.1
    df.loc[df.index[-1], "low"] = df.loc[df.index[-2], "low"] + 0.1
    df.loc[df.index[-1], "close"] = (df.loc[df.index[-1], "high"] + df.loc[df.index[-1], "low"]) / 2
    # widen prior ranges
    for i in range(n - 7, n - 1):
        df.loc[df.index[i], "high"] = df.loc[df.index[i], "close"] + 2.0
        df.loc[df.index[i], "low"] = df.loc[df.index[i], "close"] - 2.0
    out = detect_patterns(df, lookback=30, min_confidence=0.35)
    names = {p["pattern"] for p in out["patterns"]}
    assert "Inside Bar" in names
    assert "NR7" in names


def test_double_bottom_and_outputs():
    # V-shaped double bottom
    close = np.concatenate(
        [
            np.linspace(110, 100, 10),
            np.linspace(100, 108, 8),
            np.linspace(108, 100.2, 8),
            np.linspace(100.2, 109, 10),
        ]
    )
    out = detect_patterns(_ohlcv_from_close(close, noise=0.3), lookback=len(close), min_confidence=0.35)
    assert out["backend"] == PATTERN_BACKEND
    assert "attention_weights" in out
    assert out["summary"]["n_patterns"] >= 1
    top = out["patterns"][0]
    assert 0 <= top["confidence"] <= 1
    assert 0 <= top["probability"] <= 1


def test_catalog_constants():
    expected_chart = [
        "Bull Flag",
        "Bear Flag",
        "Cup Handle",
        "Ascending Triangle",
        "Descending Triangle",
        "Pennant",
        "Double Bottom",
        "Double Top",
        "Head Shoulders",
        "Inverse Head Shoulders",
        "Rectangle",
        "Wedge",
        "Channel",
        "Gap",
        "Island Reversal",
        "Volatility Squeeze",
        "NR7",
        "Inside Bar",
    ]
    assert CHART_PATTERNS == expected_chart
    assert "Hammer" in CANDLESTICK_PATTERNS
    assert "Bullish Engulfing" in CANDLESTICK_PATTERNS
    assert PATTERN_OUTPUTS == ("pattern", "confidence", "target_projection", "probability")


def test_hybrid_encoder_branches():
    rng = np.random.default_rng(0)
    window = rng.normal(size=(40, 5))
    window[:, 3] = np.cumsum(rng.normal(0, 0.01, size=40)) + 100  # close
    cnn = cnn_features(window)
    lstm = lstm_features(window)
    trans, attn = transformer_features(window)
    assert cnn.ndim == 1 and cnn.size > 0
    assert lstm.shape[0] == 8
    assert trans.size > 0
    assert abs(attn.sum() - 1.0) < 1e-5
    emb = hybrid_embedding(window)
    assert set(emb["branch_norms"]) == {"cnn", "lstm", "transformer"}


def test_pattern_api_meta():
    from fastapi.testclient import TestClient

    from services.pattern_recognition.app import app

    client = TestClient(app)
    cat = client.get("/v1/catalog")
    assert cat.status_code == 200
    body = cat.json()
    assert body["chart_patterns"] == CHART_PATTERNS
    assert body["outputs"] == list(PATTERN_OUTPUTS)

    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["hybrid_branches"] == ["cnn", "lstm", "transformer"]
