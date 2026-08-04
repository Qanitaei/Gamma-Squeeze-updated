import numpy as np
import pandas as pd

from gamma_squeeze.models.xgboost_direction import (
    DIRECTION_HORIZONS,
    train_direction_model,
)


def _toy_panel(n: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(7)
    base = pd.Timestamp("2024-01-02")
    spot = 100 + np.cumsum(rng.normal(0.05, 0.8, size=n))
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbol": "TEST",
                "as_of": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "spot": float(spot[i]),
                "ret_1d": float(rng.normal(0, 0.01)),
                "rvol_10d": float(0.15 + 0.05 * rng.random()),
                "net_gex": float(rng.normal(0, 1e5)),
                "gex_per_spot": float(rng.normal(0, 1e5)),
                "positioning_stress": float(rng.random()),
                "pcr_oi": float(0.8 + 0.4 * rng.random()),
                "regime_neg_gamma": float(rng.integers(0, 2)),
                "rsi": float(40 + 20 * rng.random()),
                "vix": float(14 + 10 * rng.random()),
            }
        )
    feat = pd.DataFrame(rows)
    # labels only for some horizons; model synthesizes the rest from spot
    labs = feat[["symbol", "as_of"]].copy()
    for h in (1, 5, 10):
        labs[f"fwd_ret_{h}d"] = feat["spot"].shift(-h) / feat["spot"] - 1.0
    cols = [
        "ret_1d",
        "rvol_10d",
        "net_gex",
        "gex_per_spot",
        "positioning_stress",
        "pcr_oi",
        "regime_neg_gamma",
        "rsi",
        "vix",
    ]
    return feat, labs, cols


def test_train_multihorizon_outputs():
    feat, labs, cols = _toy_panel()
    bundle = train_direction_model(
        feat,
        labs,
        cols,
        horizons=list(DIRECTION_HORIZONS),
        n_trials=3,
        cv_folds=3,
        include_shap=True,
    )
    assert bundle.models, "expected at least one trained horizon"
    # 20d should be synthesizable from spot
    assert 20 in bundle.models or 5 in bundle.models

    row = feat.iloc[-25]  # leave room for forward labels during train; predict any row
    pred = bundle.predict_row(row)
    assert "horizons" in pred
    for key, block in pred["horizons"].items():
        assert "classification" in block
        assert "regression" in block
        assert "probability" in block
        prob = block["probability"]
        assert set(prob) == {"down", "flat", "up"}
        assert abs(sum(prob.values()) - 1.0) < 1e-5

    # single-horizon API
    h = next(iter(bundle.models))
    one = bundle.predict_row(row, horizon=h)
    assert one["classification"] in {"down", "flat", "up"}
    assert isinstance(one["regression"], float)


def test_cv_and_optuna_metadata():
    feat, labs, cols = _toy_panel(100)
    bundle = train_direction_model(
        feat,
        labs,
        cols,
        horizons=[1, 5],
        n_trials=2,
        cv_folds=3,
        include_shap=False,
    )
    assert 1 in bundle.models
    hb = bundle.models[1]
    assert "classification_accuracy_mean" in hb.cv_scores
    assert hb.best_params
    assert "n_estimators" in hb.best_params or hb.cv_scores.get("optuna", {}).get("status")


def test_direction_horizons_constant():
    from gamma_squeeze.models.xgboost_direction import (
        DIRECTION_HORIZON_LABELS,
        DIRECTION_TARGETS,
        DIRECTION_TRAINING,
    )

    assert DIRECTION_HORIZONS == (1, 3, 5, 10, 20)
    assert DIRECTION_TARGETS == ("classification", "regression", "probability")
    assert "cross_validation" in DIRECTION_TRAINING
    assert "bayesian_optimization" in DIRECTION_TRAINING
    assert "shap_explainability" in DIRECTION_TRAINING
    assert DIRECTION_HORIZON_LABELS[1] == "1 Day Return"
    assert DIRECTION_HORIZON_LABELS[20] == "20 Day Return"


def test_direction_api_meta():
    from fastapi.testclient import TestClient

    from services.xgboost_direction.app import app

    client = TestClient(app)
    h = client.get("/v1/horizons")
    assert h.status_code == 200
    body = h.json()
    assert body["horizons"] == [1, 3, 5, 10, 20]
    assert body["targets"] == ["classification", "regression", "probability"]
    assert "bayesian_optimization" in body["training"]

    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["horizon_labels"]["5"] == "5 Day Return"


def test_zero_spot_panels_do_not_poison_targets():
    """Alpaca panels can have spot=0 rows; synthesis must not emit ±inf labels."""
    from gamma_squeeze.models.xgboost_direction import _ensure_forward_returns

    feat, labs, cols = _toy_panel(80)
    feat.loc[feat.index[:5], "spot"] = 0.0
    labs.loc[labs.index[:3], "fwd_ret_1d"] = np.inf
    labs.loc[labs.index[3:6], "fwd_ret_1d"] = np.nan
    merged = _ensure_forward_returns(feat, labs, [1, 5, 20])
    for h in (1, 5, 20):
        arr = pd.to_numeric(merged[f"fwd_ret_{h}d"], errors="coerce").to_numpy(dtype=float)
        assert not np.isinf(arr).any()
    bundle = train_direction_model(
        feat,
        labs,
        cols,
        horizons=[1, 5],
        n_trials=0,
        cv_folds=3,
        include_shap=False,
    )
    assert bundle.models, "expected training to succeed after y sanitization"
