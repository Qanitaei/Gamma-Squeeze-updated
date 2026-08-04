import numpy as np
import pandas as pd

from gamma_squeeze.alerts.channels import CHANNELS, channel_config, dispatch_alerts
from gamma_squeeze.alerts.engine import ALERT_TRIGGERS, AlertThresholds, evaluate_alert_engine


def test_trigger_and_channel_catalog():
    assert ALERT_TRIGGERS == (
        "Gamma Squeeze Probability",
        "Dealer Flip",
        "Gamma Ramp",
        "Call Wall Break",
        "Put Wall Break",
        "Large IV Expansion",
        "Large Dealer Hedge",
        "Pattern Breakout",
        "Earnings Risk",
    )
    assert CHANNELS == ("WebSocket", "Email", "SMS", "Discord", "Slack", "Webhook")


def test_evaluate_alert_engine_triggers(monkeypatch):
    from gamma_squeeze.alerts import engine as eng

    feat = pd.DataFrame(
        [
            {
                "symbol": "TEST",
                "as_of": "2025-06-02",
                "spot": 111.0,
                "positioning_stress": 0.9,
                "regime_neg_gamma": 1.0,
                "gamma_flip": 110.5,
                "call_wall": 110.0,
                "put_wall": 95.0,
                "atm_iv": 0.45,
                "rvol_10d": 0.2,
                "dealer_hedge_requirement": -8e4,
                "net_gex": -1e6,
            }
        ]
    )
    monkeypatch.setattr(eng, "load_features", lambda sym: feat)
    monkeypatch.setattr(eng, "build_features_for_symbol", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        eng,
        "load_latest_forecast",
        lambda sym: {
            "as_of": "2025-06-02",
            "horizons": [
                {
                    "horizon_days": 5,
                    "squeeze_probability": 0.72,
                    "confidence_score": 0.66,
                    "expected_magnitude_pct": 4.0,
                }
            ],
            "meta": {"composite": {"probability": 0.72, "confidence": 0.66}},
        },
    )
    monkeypatch.setattr(eng, "alpaca_configured", lambda: False)
    monkeypatch.setattr(
        eng,
        "_load_matrix",
        lambda sym, as_of: (
            {
                "underlying_price": 111.0,
                "contracts": [
                    {
                        "strike_price": 110,
                        "put_call": "call",
                        "open_interest": 1000,
                        "gamma": 0.02,
                        "delta": 0.5,
                        "volatility": 0.4,
                        "days_to_expiration": 10,
                    }
                ],
            },
            "ssd",
        ),
    )

    class _Sim:
        def to_dict(self):
            return {
                "spot": 111.0,
                "gamma_ramp": 1e5,
                "delta_hedging": -9e4,
                "dealer_position_flip": {"gamma_flip_spot": 110.8, "distance_pct": 0.002},
                "expected_hedging_volume": {"shares": 9e4},
                "meta": {"positioning": {"call_wall": 110.0, "put_wall": 95.0}},
            }

    monkeypatch.setattr(
        "gamma_squeeze.dealer.hedge_demand.simulate_dealer_hedging",
        lambda *a, **k: _Sim(),
    )
    monkeypatch.setattr(
        "services.pattern_recognition.service.run_patterns",
        lambda sym: {"patterns": [{"pattern": "Bull Flag", "confidence": 0.8}]},
    )
    monkeypatch.setattr(
        eng,
        "fetch_calendar_near",
        lambda *a, **k: [{"date": "2025-06-04", "title": "AAPL Earnings"}],
    )

    out = evaluate_alert_engine("TEST", use_live_alpaca=True, notify=False)
    types = {a["type"] for a in out["alerts"]}
    assert out["fired"] is True
    assert "gamma_squeeze_probability" in types
    assert "dealer_flip" in types
    assert "gamma_ramp" in types
    assert "call_wall_break" in types
    assert "large_iv_expansion" in types
    assert "large_dealer_hedge" in types
    assert "pattern_breakout" in types
    assert "earnings_risk" in types


def test_dispatch_skips_unconfigured():
    payload = {
        "symbol": "TEST",
        "fired": True,
        "alerts": [{"type": "x", "severity": "low", "message": "hi", "symbol": "TEST"}],
    }
    results = dispatch_alerts("TEST", payload, channels=["Email", "Discord", "Webhook"])
    assert any(r.get("skipped") for r in results)
    cfg = channel_config()
    assert "websocket" in cfg


def test_alerts_meta_endpoint(monkeypatch):
    from services.trade_alerts.app import app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        "gamma_squeeze.ingest.alpaca_api.alpaca_api_health",
        lambda: {"ok": True, "configured": True},
    )
    client = TestClient(app)
    resp = client.get("/v1/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert "Gamma Squeeze Probability" in body["notify_when"]
    assert "Discord" in body["support"]
    assert "Webhook" in body["support"]
    assert body["data_plane"] == "alpaca"
    assert "min_probability" in body["configurable"]
    assert body["default_thresholds"]["min_probability"] == 0.55
