from gamma_squeeze.features.dealer_positioning import (
    DEALER_POSITIONING_FIELDS,
    bs_greeks,
    compute_dealer_positioning,
)


def _toy_matrix(spot: float = 100.0) -> dict:
    return {
        "symbol": "TEST",
        "underlying_price": spot,
        "as_of_date": "2026-07-31",
        "contracts": [
            {
                "strike_price": 100,
                "put_call": "call",
                "open_interest": 1000,
                "total_volume": 200,
                "gamma": 0.05,
                "delta": 0.52,
                "vega": 0.12,
                "theta": -0.03,
                "volatility": 0.30,
                "days_to_expiration": 14,
            },
            {
                "strike_price": 95,
                "put_call": "put",
                "open_interest": 800,
                "total_volume": 150,
                "gamma": 0.04,
                "delta": -0.35,
                "vega": 0.10,
                "theta": -0.02,
                "volatility": 0.32,
                "days_to_expiration": 14,
            },
            {
                "strike_price": 105,
                "put_call": "call",
                "open_interest": 600,
                "total_volume": 90,
                "gamma": 0.03,
                "delta": 0.30,
                "vega": 0.09,
                "theta": -0.025,
                "volatility": 0.28,
                "days_to_expiration": 21,
            },
        ],
    }


def test_bs_greeks_call_positive_delta():
    g = bs_greeks(100, 100, 30 / 365, 0.25, is_call=True)
    assert g["delta"] > 0.4
    assert g["gamma"] > 0
    assert g["vega"] > 0


def test_dealer_positioning_fields():
    feats = compute_dealer_positioning(_toy_matrix(), as_of="2026-07-31")
    d = feats.to_dict()
    for key in DEALER_POSITIONING_FIELDS:
        assert key in d, key
    assert len(feats.gamma_by_strike) >= 2
    assert feats.call_wall is not None
    assert feats.put_wall is not None
    assert 0.0 <= feats.dealer_liquidity_score <= 1.0


def test_prior_updates_acceleration():
    m = _toy_matrix()
    first = compute_dealer_positioning(m, as_of="2026-07-30")
    m2 = _toy_matrix(spot=102.0)
    second = compute_dealer_positioning(m2, as_of="2026-07-31", prior=first)
    assert second.dealer_gamma_acceleration == second.dealer_gamma - first.dealer_gamma


def test_dealer_features_via_alpaca_adapter_mock(monkeypatch):
    """Feature Engineering dealer path uses options.alpaca (API/KV), not Schwab."""
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

    import gamma_squeeze.data_sources.registry as reg

    monkeypatch.setattr(reg, "load_all_adapters", lambda: None)
    monkeypatch.setattr(reg, "get_adapter", lambda name: _FakeAlpaca())

    out = fe_service.compute_dealer_features("AAPL", as_of="2026-07-31")
    assert out["success"] is True
    assert out["matrix_origin"] == "kv_worker"
    dp = out["dealer_positioning"]
    for key in DEALER_POSITIONING_FIELDS:
        assert key in dp
