"""Offline smoke: toy features/labels → train → schema-valid forecast."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from gamma_squeeze.config import HORIZONS
from gamma_squeeze.models.ensemble import train_ensemble_components
from gamma_squeeze.serve.schemas import validate_forecast_dict


def _toy_frames(n: int = 40):
    rows_f = []
    rows_l = []
    base = date(2025, 1, 2)
    for i in range(n):
        as_of = (base + timedelta(days=i)).isoformat()
        stress = (i % 10) / 10.0
        fragile = 1.0 if i % 3 == 0 else 0.0
        rows_f.append(
            {
                "symbol": "TEST",
                "as_of": as_of,
                "spot": 100 + i,
                "net_gex": -1e6 if fragile else 1e6,
                "call_gex": 1e6,
                "put_gex": 2e6,
                "net_dex": 0.0,
                "pcr_oi": 0.8,
                "pcr_vol": 0.9,
                "regime_neg_gamma": fragile,
                "flip_dist_pct": 0.01,
                "call_wall_dist_pct": 0.02,
                "put_wall_dist_pct": 0.03,
                "gex_per_spot": -1000.0 if fragile else 1000.0,
                "flag_fragile_gamma": fragile,
                "flag_near_flip": 1.0 if i % 5 == 0 else 0.0,
                "flag_call_crowding": 1.0 if i % 4 == 0 else 0.0,
                "flag_wall_magnet": 0.0,
                "positioning_stress": stress,
                "ret_1d": 0.01 * ((i % 5) - 2),
                "ret_5d": 0.02 * ((i % 5) - 2),
                "rvol_10d": 0.25,
                "iv_atm": 0.3,
                "skew_25d": -0.05,
                "term_slope": 0.01,
                "iv_minus_hv": 0.02,
                "call_bias": 0.1,
                "fedfunds": 5.0,
                "dgs10": 4.0,
                "t10y2y": -0.2,
            }
        )
        lab = {"symbol": "TEST", "as_of": as_of, "candidate_setup": int(stress > 0.3)}
        for h in HORIZONS:
            fwd = 0.01 * h * (1 if fragile else -0.2)
            lab[f"fwd_ret_{h}d"] = fwd
            lab[f"squeeze_{h}d"] = int(fragile and fwd > 0.02)
            lab[f"duration_label_{h}d"] = float(min(h, 3 + (i % h)))
            lab[f"magnitude_label_{h}d"] = fwd
        rows_l.append(lab)
    return pd.DataFrame(rows_f), pd.DataFrame(rows_l)


def test_train_and_infer_schema():
    features, labels = _toy_frames()
    cols = [
        c
        for c in features.columns
        if c not in ("symbol", "as_of")
    ]
    ens = train_ensemble_components(features, labels, cols)
    row = features.iloc[-1]
    matrix = {
        "symbol": "TEST",
        "underlying_price": float(row["spot"]),
        "contracts": [
            {
                "strike_price": float(row["spot"]),
                "put_call": "call",
                "open_interest": 500,
                "gamma": 0.02,
                "delta": 0.5,
                "volatility": 0.3,
                "days_to_expiration": 10,
            }
        ],
    }
    fc = ens.predict(row, symbol="TEST", as_of=str(row["as_of"]), matrix=matrix)
    errors = validate_forecast_dict(fc.to_dict())
    assert errors == [], errors
    assert len(fc.dealer_hedging_demand) >= 3
    assert fc.trade_recommendations
    assert fc.explanations
