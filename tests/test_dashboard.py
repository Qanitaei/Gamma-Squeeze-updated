import numpy as np
import pandas as pd

from gamma_squeeze.viz.institutional_dashboard import (
    DASHBOARD_PANELS,
    HEATMAPS,
    _surface_grids,
    build_institutional_dashboard,
)


def test_catalog():
    assert "Market Regime" in DASHBOARD_PANELS
    assert "Trade Journal" in DASHBOARD_PANELS
    assert "3D Gamma Surface" in HEATMAPS
    assert len(DASHBOARD_PANELS) == 19
    assert len(HEATMAPS) == 6


def test_surface_grids_from_contracts():
    contracts = []
    for dte in (7, 14, 30):
        for strike in (95, 100, 105):
            for side, pc in ((1, "call"), (-1, "put")):
                contracts.append(
                    {
                        "strike_price": strike,
                        "days_to_expiration": dte,
                        "put_call": pc,
                        "open_interest": 100 * strike / 10,
                        "total_volume": 50,
                        "gamma": 0.02,
                        "implied_volatility": 0.25 + dte / 1000,
                    }
                )
    surf = _surface_grids(contracts, spot=100.0)
    assert surf["strikes"]
    assert surf["dtes"]
    assert surf["iv"]
    assert surf["strike_distribution"]["total_oi"]


def test_build_dashboard_offline(monkeypatch):
    from gamma_squeeze.viz import institutional_dashboard as idash

    def fake_features(sym, **kwargs):
        n = 40
        rows = []
        for i in range(n):
            rows.append(
                {
                    "symbol": sym,
                    "as_of": f"2025-05-{(i % 28) + 1:02d}",
                    "spot": 100 + i * 0.1,
                    "ret_1d": float(np.sin(i / 5) * 0.01),
                    "positioning_stress": 0.6,
                    "regime_neg_gamma": 1.0,
                    "net_gex": -3e5,
                    "gamma_flip": 98.0,
                    "call_wall": 110.0,
                    "put_wall": 95.0,
                    "atm_iv": 0.3,
                    "rvol_10d": 0.22,
                    "dealer_delta": -10000,
                    "dealer_gamma": -500,
                    "dealer_vega": 20,
                    "dealer_theta": -10,
                    "vix": 20.0,
                    "fed_funds": 4.0,
                    "yield_spread": 0.3,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(idash, "load_features", fake_features)
    monkeypatch.setattr(idash, "build_features_for_symbol", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(idash, "load_latest_forecast", lambda sym: None)
    monkeypatch.setattr(idash, "fetch_calendar_near", lambda *a, **k: [])
    monkeypatch.setattr(idash, "alpaca_configured", lambda: False)
    monkeypatch.setattr(
        idash,
        "_load_alpaca_matrix",
        lambda sym, as_of: (
            {
                "underlying_price": 100.0,
                "contracts": [
                    {
                        "strike_price": 100,
                        "days_to_expiration": 14,
                        "put_call": "call",
                        "open_interest": 200,
                        "total_volume": 80,
                        "gamma": 0.03,
                        "implied_volatility": 0.28,
                    }
                ],
            },
            "ssd",
        ),
    )

    dash = build_institutional_dashboard("TEST", use_live_alpaca=True, include_pipeline_extras=False)
    assert dash["symbol"] == "TEST"
    assert "market_regime" in dash["panels"]
    assert "forecast" in dash["panels"]
    assert "volatility_surface" in dash["heatmaps"]
    assert "gamma_surface_3d" in dash["plotly"]
    assert "html" not in dash


def test_dashboard_meta_uses_alpaca(monkeypatch):
    from gamma_squeeze.viz import institutional_dashboard as idash

    monkeypatch.setattr(
        "gamma_squeeze.ingest.alpaca_api.alpaca_api_health",
        lambda: {"ok": True, "configured": True, "paper": "true"},
    )
    meta = idash.dashboard_meta()
    assert meta["data_plane"] == "alpaca"
    assert meta["alpaca"]["configured"] is True
    assert "Market Regime" in meta["panels"]
