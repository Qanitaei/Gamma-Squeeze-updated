import pandas as pd

from gamma_squeeze.features.options_metrics import (
    OPTIONS_METRICS_FIELDS,
    compute_options_metrics,
    iv_rank_percentile,
    probability_itm_call,
    probability_touch,
    risk_neutral_density_from_calls,
)


def _toy_matrix(spot: float = 100.0) -> dict:
    contracts = []
    for strike in (90, 95, 100, 105, 110):
        for side, dte in (("call", 14), ("put", 14)):
            moneyness = strike / spot
            iv = 0.25 + 0.08 * abs(moneyness - 1.0)
            contracts.append(
                {
                    "strike_price": strike,
                    "put_call": side,
                    "open_interest": 500 + strike,
                    "total_volume": 50 + (strike % 10) * 3,
                    "volatility": iv,
                    "days_to_expiration": dte,
                    "expiration_date": "2026-08-15",
                    "mid": max(spot - strike, 1.0) if side == "call" else max(strike - spot, 1.0),
                }
            )
    for strike in (95, 100, 105):
        contracts.append(
            {
                "strike_price": strike,
                "put_call": "call",
                "open_interest": 200,
                "total_volume": 20,
                "volatility": 0.28,
                "days_to_expiration": 45,
                "expiration_date": "2026-09-19",
            }
        )
    return {
        "symbol": "TEST",
        "underlying_price": spot,
        "as_of_date": "2026-07-31",
        "contracts": contracts,
    }


def test_options_metrics_locked_fields():
    ohlcv = pd.DataFrame(
        {"close": [100 + i * 0.2 for i in range(60)]},
        index=pd.date_range("2026-05-01", periods=60, freq="B").strftime("%Y-%m-%d"),
    )
    feats = compute_options_metrics(
        _toy_matrix(),
        as_of="2026-07-31",
        ohlcv=ohlcv,
        prior_oi=5000,
        atm_iv_history=[0.20, 0.22, 0.30, 0.27],
    )
    d = feats.to_dict()
    for key in OPTIONS_METRICS_FIELDS:
        assert key in d, key
    assert feats.open_interest > 0
    assert feats.put_call_ratio > 0
    assert feats.atm_iv is not None
    assert feats.expected_move is not None
    assert isinstance(feats.risk_neutral_density, list)
    assert feats.risk_neutral_density


def test_iv_rank_percentile():
    rank, pct = iv_rank_percentile([0.1, 0.2, 0.3, 0.4], 0.3)
    assert rank is not None and 0 <= rank <= 1
    assert pct is not None and 0 <= pct <= 1


def test_prob_helpers():
    assert 0 < probability_itm_call(100, 100, 30 / 365, 0.25) < 1
    assert 0 < probability_touch(100, 105, 30 / 365, 0.25) <= 1


def test_rnd_nonnegative():
    strikes = [90, 95, 100, 105, 110]
    calls = [12.0, 8.0, 5.0, 3.0, 1.5]
    dens = risk_neutral_density_from_calls(100, strikes, calls, 30 / 365)
    assert dens
    assert all(row["density"] >= 0 for row in dens)


def test_options_metrics_via_alpaca_adapter_mock(monkeypatch):
    from services.feature_engineering import service as fe_service

    matrix = _toy_matrix()
    matrix["symbol"] = "AAPL"

    class _FakeAlpaca:
        def fetch_matrix(self, symbol, as_of=None):
            return {
                "success": True,
                "as_of": "2026-07-31",
                "data": {"matrix": matrix},
                "meta": {"origin": "kv_worker"},
            }

    monkeypatch.setattr(fe_service, "list_local_dates", lambda *a, **k: [])
    monkeypatch.setattr(fe_service, "load_local_matrix", lambda *a, **k: None)
    monkeypatch.setattr(fe_service, "fetch_daily_ohlcv", lambda *a, **k: pd.DataFrame())

    import gamma_squeeze.data_sources.registry as reg

    monkeypatch.setattr(reg, "load_all_adapters", lambda: None)
    monkeypatch.setattr(reg, "get_adapter", lambda name: _FakeAlpaca())

    out = fe_service.compute_options_metrics_features("AAPL", as_of="2026-07-31")
    assert out["success"] is True
    assert out["matrix_origin"] == "kv_worker"
    om = out["options_metrics"]
    for key in OPTIONS_METRICS_FIELDS:
        assert key in om
