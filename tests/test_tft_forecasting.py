import numpy as np
import pandas as pd

from gamma_squeeze.deep.temporal_model import (
    TFT_HORIZON_LABELS,
    TFT_HORIZONS,
    TFT_OUTPUTS,
    TFT_QUANTILES,
    TFT_TARGET_LABELS,
    TFT_TARGETS,
    TFTForecastModel,
)


def _toy_features(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2024-06-03")
    spot = 100 + np.cumsum(rng.normal(0.05, 0.7, size=n))
    rows = []
    for i in range(n):
        gex = float(rng.normal(0, 2e5))
        rows.append(
            {
                "symbol": "TEST",
                "as_of": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "spot": float(spot[i]),
                "ret_1d": float(rng.normal(0, 0.01)),
                "ret_5d": float(rng.normal(0, 0.02)),
                "rvol_10d": float(0.18 + 0.04 * rng.random()),
                "net_gex": gex,
                "gex_per_spot": gex,
                "gamma_flip": float(spot[i] * (0.98 + 0.01 * rng.random())),
                "zero_gamma": float(spot[i] * 0.99),
                "call_wall": float(spot[i] * 1.05),
                "put_wall": float(spot[i] * 0.95),
                "dealer_hedge_requirement": float(rng.normal(0, 1e4)),
                "dealer_gamma": float(rng.normal(0, 1e3)),
                "positioning_stress": float(rng.random()),
                "pcr_oi": float(0.9 + 0.2 * rng.random()),
                "atm_iv": float(0.22 + 0.05 * rng.random()),
                "iv_atm": float(0.22 + 0.05 * rng.random()),
                "expected_move": float(spot[i] * 0.02),
                "rsi": float(45 + 10 * rng.random()),
                "vix": float(15 + 5 * rng.random()),
            }
        )
    return pd.DataFrame(rows)


def test_tft_fit_forecast_quantiles():
    feat = _toy_features(110)
    model = TFTForecastModel(
        lookback=16,
        horizons=[1, 2, 5],
        targets=list(TFT_TARGETS),
        quantile_mode="median_spread",
    )
    metrics = model.fit(feat, pd.DataFrame())
    assert metrics["status"] == "ok"
    out = model.forecast(feat)
    assert "attention_weights" in out
    assert out["attention_weights"]["temporal"]
    assert out["attention_weights"]["variables"]
    assert "attention_weights" in out["outputs"]
    for h in ("1d", "2d", "5d"):
        assert h in out["horizons"]
        hz = out["horizons"][h]
        assert "horizon_label" in hz
        for target in TFT_TARGETS:
            block = hz[target]
            for key in ("median", "median_forecast", "q10", "q25", "q75", "q90"):
                assert key in block
                assert block[key] is not None
            pi = block["prediction_interval"]
            assert pi["low"] <= pi["high"]
            assert block["q10"] <= block["median"] <= block["q90"]
            assert block["label"] == TFT_TARGET_LABELS[target]


def test_tft_full_quantile_mode_smoke():
    feat = _toy_features(90)
    model = TFTForecastModel(
        lookback=12,
        horizons=[1, 5],
        targets=["future_price", "net_gex", "expected_move"],
        quantile_mode="full",
    )
    metrics = model.fit(feat, pd.DataFrame())
    assert metrics["status"] == "ok"
    out = model.forecast(feat)
    price = out["horizons"]["5d"]["future_price"]
    assert price["prediction_interval"]["coverage"] == "80pct_q10_q90"


def test_tft_constants():
    assert TFT_HORIZONS == (1, 2, 3, 5, 10)
    assert TFT_QUANTILES == (0.10, 0.25, 0.50, 0.75, 0.90)
    assert TFT_TARGETS == (
        "future_price",
        "iv",
        "net_gex",
        "dealer_hedge_requirement",
        "gamma_flip",
        "call_wall",
        "put_wall",
        "expected_move",
    )
    assert TFT_TARGET_LABELS["future_price"] == "Future Price"
    assert TFT_TARGET_LABELS["dealer_hedge_requirement"] == "Dealer Hedge Requirement"
    assert TFT_HORIZON_LABELS[10] == "10 Trading Days"
    assert "prediction_interval" in TFT_OUTPUTS
    assert "attention_weights" in TFT_OUTPUTS


def test_tft_api_meta():
    from fastapi.testclient import TestClient

    from services.tft_forecasting.app import app

    client = TestClient(app)
    h = client.get("/v1/horizons")
    assert h.status_code == 200
    body = h.json()
    assert body["horizons"] == [1, 2, 3, 5, 10]
    assert body["target_labels"]["net_gex"] == "Net GEX"
    assert "median_forecast" in body["outputs"]

    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["horizon_labels"]["5"] == "5 Trading Days"


def test_tft_all_default_horizons():
    feat = _toy_features(120)
    model = TFTForecastModel(lookback=16, quantile_mode="median_spread")
    metrics = model.fit(feat, pd.DataFrame())
    assert metrics["status"] == "ok"
    out = model.forecast(feat)
    for h in TFT_HORIZONS:
        assert f"{h}d" in out["horizons"]
        assert set(TFT_TARGETS).issubset(set(out["horizons"][f"{h}d"]["targets"]))
