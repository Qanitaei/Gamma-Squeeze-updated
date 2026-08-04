import numpy as np
import pandas as pd

from gamma_squeeze.features.technical_indicators import (
    TECHNICAL_INDICATOR_COLUMNS,
    TECHNICAL_INDICATORS_FIELDS,
    compute_technical_indicators,
)


def _toy_ohlcv(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-12-01", periods=n, freq="B").strftime("%Y-%m-%d")
    close = 100 + np.cumsum(rng.normal(0, 0.8, size=n))
    high = close + rng.uniform(0.2, 1.5, size=n)
    low = close - rng.uniform(0.2, 1.5, size=n)
    open_ = close + rng.normal(0, 0.3, size=n)
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_technical_indicators_locked_catalog():
    ohlcv = _toy_ohlcv()
    feats = compute_technical_indicators(ohlcv, symbol="TEST", as_of=str(ohlcv.index[-1]))
    catalog = feats.catalog_dict()
    for key in TECHNICAL_INDICATORS_FIELDS:
        assert key in catalog, key
        assert catalog[key] is not None, key

    assert 0 <= feats.rsi <= 100
    assert 0 <= feats.money_flow_index <= 100
    assert feats.supertrend_dir in (-1.0, 1.0)
    assert isinstance(catalog["volume_profile"]["bins"], list)
    assert catalog["volume_profile"]["bins"]
    assert 0 <= feats.liquidity_score <= 1


def test_flat_features_excludes_curves():
    feats = compute_technical_indicators(_toy_ohlcv(80), symbol="TEST")
    flat = feats.flat_features()
    assert "volume_profile" not in flat
    assert "meta" not in flat
    for col in TECHNICAL_INDICATOR_COLUMNS:
        assert col in flat


def test_anchored_vwap_respects_anchor():
    ohlcv = _toy_ohlcv(60)
    anchor = str(ohlcv.index[30])
    feats = compute_technical_indicators(ohlcv, symbol="TEST", anchor_date=anchor)
    assert feats.anchored_vwap is not None
    assert feats.meta["anchor_idx"] == 30


def test_breadth_passthrough():
    ohlcv = _toy_ohlcv(40)
    feats = compute_technical_indicators(
        ohlcv,
        symbol="TEST",
        breadth={
            "tick_index": 12.5,
            "advance_decline": 400.0,
            "trin": 0.85,
            "breadth": 0.62,
        },
    )
    assert feats.tick_index == 12.5
    assert feats.advance_decline == 400.0
    assert feats.trin == 0.85
    assert feats.breadth == 0.62


def test_technical_features_service(monkeypatch):
    from services.feature_engineering import service as fe_service

    ohlcv = _toy_ohlcv()
    monkeypatch.setattr(fe_service, "fetch_daily_ohlcv", lambda *a, **k: ohlcv)
    out = fe_service.compute_technical_features("TEST")
    assert out["success"] is True
    ti = out["technical_indicators"]
    for key in TECHNICAL_INDICATORS_FIELDS:
        assert key in ti
    assert out["feature_names"] == list(TECHNICAL_INDICATORS_FIELDS)
