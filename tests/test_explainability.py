import numpy as np
import pandas as pd

from gamma_squeeze.explain.engine import (
    EXPLAIN_COMPONENTS,
    compute_counterfactuals,
    compute_model_confidence,
    compute_partial_dependence,
    compute_shap,
    explain_prediction,
)
from gamma_squeeze.models.ensemble import SqueezeEnsemble
from gamma_squeeze.serve.schemas import HorizonForecast, MagnitudeQuantiles


def _row() -> dict:
    return {
        "symbol": "TEST",
        "as_of": "2025-06-02",
        "positioning_stress": 0.8,
        "regime_neg_gamma": 1.0,
        "flag_fragile_gamma": 1.0,
        "net_gex": -4e5,
        "gex_per_spot": -4e5,
        "pcr_oi": 1.1,
        "atm_iv": 0.33,
        "rvol_10d": 0.28,
        "ret_5d": 0.02,
        "iv_minus_hv": 0.05,
        "call_bias": 0.2,
    }


def _features(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        r = _row()
        r["as_of"] = f"2025-05-{i+1:02d}" if i < 28 else "2025-06-02"
        r["ret_1d"] = float(np.sin(i / 3) * 0.01)
        r["positioning_stress"] = 0.4 + 0.02 * i
        rows.append(r)
    return pd.DataFrame(rows)


def test_components_catalog():
    assert EXPLAIN_COMPONENTS == (
        "SHAP",
        "Attention Maps",
        "Feature Importance",
        "Counterfactual Analysis",
        "Partial Dependence",
        "Decision Trace",
        "Model Confidence",
    )


def test_shap_library_available():
    from gamma_squeeze.explain.engine import HAS_SHAP

    assert HAS_SHAP, "shap required for Phase 13 (pip install -r requirements-explain.txt)"


def test_explain_prediction_bundle():
    feat = _features()
    row = feat.iloc[-1]
    bundle = explain_prediction(
        "TEST",
        row,
        features=feat,
        prediction={"probability": 0.72, "magnitude": 4.0, "confidence": 0.65, "horizon_days": 5},
    )
    d = bundle.to_dict()
    assert d["why"]
    assert d["shap"]["values"]
    assert d["attention_maps"]["temporal"]
    assert d["feature_importance"]
    assert d["counterfactuals"]
    assert d["partial_dependence"]["curves"]
    assert d["decision_trace"][-1]["stage"] == "decision"
    assert 0 <= d["model_confidence"]["score"] <= 1
    assert d["model_confidence"]["rating"] in {"High", "Medium", "Low", "Very Low"}
    for key in (
        "shap",
        "attention_maps",
        "feature_importance",
        "counterfactuals",
        "partial_dependence",
        "decision_trace",
        "model_confidence",
    ):
        assert key in d


def test_ensemble_attaches_why():
    feat = _features()
    row = feat.iloc[-1]
    fc = SqueezeEnsemble.load_latest().predict(row, symbol="TEST", as_of="2025-06-02", matrix=None)
    assert fc.meta.get("why")
    assert fc.meta.get("explainability", {}).get("decision_trace")
    assert fc.explanations
    assert fc.model_versions.get("explainability")


def test_shap_and_counterfactual_helpers():
    row = _row()
    cols = ["positioning_stress", "net_gex", "atm_iv", "rvol_10d"]
    bg = _features()[cols]
    shap_block = compute_shap(row, bg, cols, predict_fn=None)
    assert shap_block["backend"] == "linear_attribution"
    cfs = compute_counterfactuals(row, cols, None, base_pred=0.3)
    assert cfs and "statement" in cfs[0]
    pdp = compute_partial_dependence(row, bg, cols, None)
    assert "positioning_stress" in pdp["curves"]
    conf = compute_model_confidence({"probability": 0.6, "confidence": 0.7}, row, shap_block)
    assert conf["score"] >= 0


def test_explainability_service_meta(monkeypatch):
    from services.explainability import service as expl_svc
    from services.explainability.app import app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        expl_svc,
        "alpaca_api_health",
        lambda: {"ok": True, "configured": True, "trading_base": "https://paper-api.alpaca.markets"},
    )
    client = TestClient(app)
    resp = client.get("/v1/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert "SHAP" in body["provide"]
    assert body["rule"] == "Every prediction must explain WHY it was produced"
    assert body["data_plane"] == "alpaca"
    assert body["alpaca"]["configured"] is True
    assert body["provide_keys"]["SHAP"] == "shap"
    assert body["has_shap"] is True