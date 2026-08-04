import numpy as np

from gamma_squeeze.metrics.classification import classification_metrics
from gamma_squeeze.metrics.regression import mae, regression_metrics, rmse
from gamma_squeeze.metrics.risk import expected_shortfall, risk_metrics, tail_risk
from gamma_squeeze.metrics.tracker import TRACKED_METRICS, compute_from_arrays, metrics_meta
from gamma_squeeze.metrics.trading import (
    annual_return,
    average_hold_time,
    average_trade,
    calmar_ratio,
    maximum_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    trading_metrics,
    win_rate,
)


def test_tracked_catalog():
    assert len(TRACKED_METRICS) == 20
    for name in (
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC",
        "Brier Score",
        "Log Loss",
        "MAE",
        "RMSE",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Maximum Drawdown",
        "Annual Return",
        "Win Rate",
        "Profit Factor",
        "Average Trade",
        "Average Hold Time",
        "Tail Risk",
        "Expected Shortfall",
    ):
        assert name in TRACKED_METRICS


def test_classification_bundle():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3, 0.6, 0.4])
    m = classification_metrics(y, proba=p, threshold=0.5)
    assert m["status"] == "ok"
    assert 0.0 <= m["Accuracy"] <= 1.0
    assert 0.0 <= m["Precision"] <= 1.0
    assert 0.0 <= m["Recall"] <= 1.0
    assert 0.0 <= m["F1"] <= 1.0
    assert m["ROC AUC"] >= 0.5
    assert 0.0 <= m["Brier Score"] <= 1.0
    assert m["Log Loss"] > 0.0


def test_regression_mae_rmse():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.5, 2.0, 2.5])
    assert abs(mae(yt, yp) - (0.5 + 0.0 + 0.5) / 3) < 1e-9
    assert abs(rmse(yt, yp) - np.sqrt((0.25 + 0 + 0.25) / 3)) < 1e-9
    m = regression_metrics(yt, yp)
    assert m["MAE"] == mae(yt, yp)
    assert m["RMSE"] == rmse(yt, yp)


def test_trading_and_risk():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=252)
    rets[::10] = -0.03
    m = trading_metrics(rets)
    assert m["n_periods"] == 252
    assert m["Sharpe Ratio"] is not None
    assert m["Sortino Ratio"] is not None
    assert 0.0 <= m["Maximum Drawdown"] <= 1.0
    assert isinstance(m["Annual Return"], float)
    assert 0.0 <= m["Win Rate"] <= 1.0
    assert m["Average Trade"] is not None
    assert m["Average Hold Time"] >= 1.0

    assert sharpe_ratio(rets) == m["Sharpe Ratio"]
    assert sortino_ratio(rets) == m["Sortino Ratio"]
    assert abs(calmar_ratio(rets) - (annual_return(rets) / maximum_drawdown(returns=rets))) < 1e-9 or m[
        "Calmar Ratio"
    ] is None
    assert win_rate(rets[rets != 0]) >= 0.0
    assert abs(profit_factor(np.array([0.1, -0.05, 0.2, -0.05])) - 3.0) < 1e-9
    assert average_trade(np.array([0.1, -0.1])) == 0.0
    assert average_hold_time(np.array([2.0, 4.0])) == 3.0

    rk = risk_metrics(rets, alpha=0.05)
    assert rk["Tail Risk"] >= 0.0
    assert rk["Expected Shortfall"] >= rk["Tail Risk"] - 1e-9
    assert expected_shortfall(rets) == rk["Expected Shortfall"]
    assert tail_risk(rets) == rk["Tail Risk"]


def test_compute_from_arrays_flat():
    report = compute_from_arrays(
        y_true=np.array([0, 1, 1, 0, 1]),
        proba=np.array([0.1, 0.9, 0.8, 0.2, 0.7]),
        y_reg_true=np.array([1.0, 2.0, 1.5, 0.5, 2.5]),
        y_reg_pred=np.array([1.1, 1.8, 1.4, 0.7, 2.0]),
        returns=np.array([0.01, -0.02, 0.015, -0.005, 0.02]),
        signal=np.array([1, 1, 0, 1, 1]),
        symbols=["TEST"],
    )
    assert report.status == "ok"
    flat = report.flat
    for name in TRACKED_METRICS:
        assert name in flat
    assert metrics_meta()["service"] == "performance_metrics"
    assert len(metrics_meta()["track"]) == 20


def test_service_endpoints(monkeypatch):
    from fastapi.testclient import TestClient

    from services.performance_metrics import service as svc
    from services.performance_metrics.app import app

    monkeypatch.setattr(
        svc,
        "track_symbol_metrics",
        lambda *a, **k: type(
            "R",
            (),
            {
                "to_dict": lambda self: {
                    "symbols": ["AAPL"],
                    "status": "ok",
                    "classification": {"Accuracy": 0.7},
                    "regression": {"MAE": 0.1},
                    "trading": {"Sharpe Ratio": 1.2},
                    "risk": {"Tail Risk": 0.03},
                    "tracked": list(TRACKED_METRICS),
                    "flat": {"Accuracy": 0.7, "Sharpe Ratio": 1.2},
                }
            },
        )(),
    )
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert "Accuracy" in client.get("/v1/meta").json()["track"]
    resp = client.post("/v1/track", json={"symbols": ["AAPL"]})
    assert resp.status_code == 200
    assert resp.json()["data"]["status"] == "ok"

    comp = client.post(
        "/v1/compute",
        json={
            "y_true": [0, 1, 1, 0],
            "proba": [0.2, 0.8, 0.7, 0.3],
            "returns": [0.01, -0.01, 0.02, 0.0],
        },
    )
    assert comp.status_code == 200
    assert comp.json()["data"]["status"] == "ok"
    assert "ROC AUC" in comp.json()["data"]["flat"]
