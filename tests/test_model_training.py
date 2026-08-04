import numpy as np

from gamma_squeeze.training.auto_retrain import AutoRetrainPolicy, decide_retrain
from gamma_squeeze.training.drift import (
    detect_all_drift,
    detect_concept_drift,
    detect_feature_drift,
    detect_prediction_drift,
)
from gamma_squeeze.training.hyperopt import bayesian_search, fit_best_classifier
from gamma_squeeze.training.pipeline import TRAINING_CAPABILITIES, training_meta
from gamma_squeeze.training.validation import (
    evaluate_splits,
    ranking_auc,
    time_series_splits,
    walk_forward_splits,
)
from gamma_squeeze.training.versioning import next_version, save_versioned_model


def test_capabilities_catalog():
    assert TRAINING_CAPABILITIES == (
        "Walk Forward Validation",
        "Time Series Split",
        "Bayesian Hyperparameter Search",
        "Early Stopping",
        "MLflow Tracking",
        "Version Control",
        "Automatic Retraining",
        "Drift Detection",
        "Feature Drift",
        "Prediction Drift",
        "Concept Drift",
    )


def test_walk_forward_and_tss():
    wf = walk_forward_splits(100, n_splits=4, expanding=True)
    assert len(wf) >= 2
    for tr, te in wf:
        assert tr[-1] < te[0]
    tss = time_series_splits(80, n_splits=3)
    assert len(tss) == 3


def test_evaluate_splits_auc():
    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)

    def fit_predict(Xtr, ytr, Xte):
        # simple linear score
        w = np.linalg.lstsq(Xtr, ytr.astype(float), rcond=None)[0]
        return Xte @ w

    report = evaluate_splits(X, y, fit_predict, method="walk_forward", n_splits=3)
    assert report.aggregate["n_folds"] >= 2
    assert 0.0 <= report.aggregate["auc_mean"] <= 1.0


def test_bayesian_search_and_early_stop():
    import optuna  # noqa: F401 — Phase 16 requires optuna (requirements-train.txt)

    rng = np.random.default_rng(1)
    n = 80
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] > 0).astype(int)
    # ensure both classes
    y[0], y[1] = 0, 1
    result = bayesian_search(X, y, n_trials=6, n_splits=2, patience_no_improve=3, early_stopping_rounds=5)
    assert result.status == "ok"
    assert result.best_params
    assert result.sampler == "TPESampler"
    model = fit_best_classifier(X, y, result.best_params)
    assert model is not None
    proba = model.predict_proba(X)[:, 1]
    assert ranking_auc(y, proba) >= 0.5


def test_feature_prediction_concept_drift():
    rng = np.random.default_rng(2)
    X_ref = rng.normal(size=(200, 5))
    X_cur = rng.normal(loc=1.5, size=(200, 5))  # shifted
    feat = detect_feature_drift(X_ref, X_cur, [f"f{i}" for i in range(5)], psi_threshold=0.1)
    assert feat["triggered"]

    scores_ref = rng.uniform(0.2, 0.4, size=200)
    scores_cur = rng.uniform(0.6, 0.9, size=200)
    pred = detect_prediction_drift(scores_ref, scores_cur, ks_threshold=0.15)
    assert pred["triggered"]

    y_ref = (scores_ref > 0.3).astype(int)
    y_cur = (scores_cur < 0.7).astype(int)  # concept change vs scores
    concept = detect_concept_drift(y_ref, scores_ref, y_cur, scores_cur, auc_drop_threshold=0.05)
    assert "auc_drop" in concept

    report = detect_all_drift(
        X_ref=X_ref,
        X_cur=X_cur,
        scores_ref=scores_ref,
        scores_cur=scores_cur,
        y_ref=y_ref,
        y_cur=y_cur,
    )
    assert report.triggered
    assert "feature_drift" in report.reasons


def test_auto_retrain_policy():
    drift = {
        "feature_drift": {"triggered": True},
        "prediction_drift": {"triggered": False},
        "concept_drift": {"triggered": False},
        "triggered": True,
        "reasons": ["feature_drift"],
    }
    d = decide_retrain(drift, policy=AutoRetrainPolicy())
    assert d.should_retrain
    assert "feature_drift" in d.reasons

    d2 = decide_retrain(drift, policy=AutoRetrainPolicy(on_feature_drift=False))
    assert not d2.should_retrain

    d3 = decide_retrain(drift, policy=AutoRetrainPolicy(force=True, on_feature_drift=False))
    assert d3.should_retrain


def test_version_control(tmp_path, monkeypatch):
    from gamma_squeeze.training import versioning as ver

    monkeypatch.setattr(ver, "resolve_models_root", lambda: tmp_path)
    monkeypatch.setattr("gamma_squeeze.retrain.registry.resolve_models_root", lambda: tmp_path)

    model = fit_best_classifier(
        np.array([[0.0], [1.0], [0.2], [0.8]]),
        np.array([0, 1, 0, 1]),
        {},
    )
    assert model is not None
    v1 = save_versioned_model(model, name="squeeze_probability", backend="test", metrics={"auc": 0.7})
    assert v1.version.startswith("v")
    assert (tmp_path / "versions" / "squeeze_probability" / v1.version / "model.joblib").is_file()
    v2 = next_version("squeeze_probability", root=tmp_path)
    assert v2 != v1.version


def test_training_meta():
    meta = training_meta()
    assert meta["service"] == "model_training"
    assert "Walk Forward Validation" in meta["implement"]
    assert "Feature Drift" in meta["drift"]
    assert meta["hyperopt"]["available"] is True
    assert "/v1/drift" in meta["endpoints"]


def test_service_train_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from services.model_training import service as svc
    from services.model_training.app import app

    class _R:
        def to_dict(self):
            return {
                "symbols": ["AAPL"],
                "status": "ok",
                "metrics": {"n_rows": 10},
                "validation": {},
                "hyperopt": {},
                "drift": {},
                "retrain_decision": {},
                "version": {},
                "mlflow": {},
            }

    monkeypatch.setattr(svc, "run_model_training", lambda *a, **k: _R())
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/meta").status_code == 200
    resp = client.post("/v1/train", json={"symbols": ["AAPL"], "n_trials": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "model_training"
    assert body["data"]["status"] == "ok"
