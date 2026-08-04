"""Unit tests for ensemble confidence blending + schema completeness."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from gamma_squeeze.config import HORIZONS
from gamma_squeeze.models.ensemble import (
    blend_confidence,
    _direction_signal,
    _regime_signal,
)
from gamma_squeeze.serve.schemas import validate_forecast_dict


def test_blend_confidence_bounds_and_weights():
    row = {"positioning_stress": 0.7}
    low = blend_confidence(
        0.2,
        row,
        regime={"agreement": 0.2, "source": "hmm_artifact"},
        direction={"agreement": 0.2, "source": "xgb_artifact"},
    )
    high = blend_confidence(
        0.9,
        row,
        regime={"agreement": 0.95, "source": "hmm_artifact"},
        direction={"agreement": 0.9, "source": "xgb_artifact"},
    )
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_blend_confidence_graceful_fallback():
    row = {"positioning_stress": 0.6, "regime_neg_gamma": 1.0, "ret_5d": 0.02}
    # Missing artifacts → fallback sources; still returns valid score
    score = blend_confidence(
        0.55,
        row,
        regime={"agreement": 0.5, "source": "fallback"},
        direction={"agreement": 0.5, "source": "fallback"},
    )
    assert 0.0 <= score <= 1.0
    # Fallback signals themselves should not raise
    reg = _regime_signal(row)
    direc = _direction_signal(row)
    assert "agreement" in reg and "source" in reg
    assert "agreement" in direc and "source" in direc
    blended = blend_confidence(0.55, row, regime=reg, direction=direc)
    assert 0.0 <= blended <= 1.0


def test_regime_agreement_higher_for_squeeze_states():
    row = {"positioning_stress": 0.8, "regime_neg_gamma": 1.0}
    # Heuristic fallback with neg gamma should lean squeeze-friendly
    reg = _regime_signal(row)
    assert reg["source"] in ("fallback", "hmm_artifact")
    assert reg["agreement"] >= 0.4


def test_predict_schema_includes_confidence_meta():
    rows_f = []
    rows_l = []
    base = date(2025, 1, 2)
    for i in range(36):
        as_of = (base + timedelta(days=i)).isoformat()
        fragile = 1.0 if i % 3 == 0 else 0.0
        stress = (i % 10) / 10.0
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

    features = pd.DataFrame(rows_f)
    labels = pd.DataFrame(rows_l)
    cols = [c for c in features.columns if c not in ("symbol", "as_of")]
    from gamma_squeeze.models.ensemble import train_ensemble_components

    ens = train_ensemble_components(features, labels, cols)
    row = features.iloc[-1]
    fc = ens.predict(row, symbol="TEST", as_of=str(row["as_of"]), feature_panel=features)
    payload = fc.to_dict()
    errors = validate_forecast_dict(payload)
    assert errors == [], errors
    assert "hmm_regime" in fc.model_versions
    assert "xgboost_direction" in fc.model_versions
    assert "regime" in fc.meta
    assert "direction" in fc.meta
    for h in fc.horizons:
        assert 0.0 <= h.confidence_score <= 1.0
        assert h.return_distribution
        assert "p10" in h.magnitude_quantiles.__dict__
    assert fc.trade_recommendations
    assert fc.explanations
