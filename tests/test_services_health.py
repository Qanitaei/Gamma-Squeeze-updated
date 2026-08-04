"""Each microservice app exposes /health independently."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

SERVICE_MODULES = [
    "services.data_collection.app",
    "services.feature_engineering.app",
    "services.pattern_recognition.app",
    "services.regime_hmm.app",
    "services.xgboost_direction.app",
    "services.tft_forecasting.app",
    "services.dealer_hedging.app",
    "services.gamma_squeeze_engine.app",
    "services.ppo_agent.app",
    "services.portfolio_hedging.app",
    "services.dashboard.app",
    "services.trade_alerts.app",
    "services.explainability.app",
    "services.model_training.app",
    "services.performance_metrics.app",
    "services.orchestrator.app",
]


@pytest.mark.parametrize("module_name", SERVICE_MODULES)
def test_service_health(module_name: str):
    mod = importlib.import_module(module_name)
    client = TestClient(mod.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"]
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert "ready" in ready.json()


def test_orchestrator_registry():
    from services.orchestrator.app import app

    client = TestClient(app)
    resp = client.get("/v1/registry")
    assert resp.status_code == 200
    payload = resp.json()
    services = payload["services"]
    assert len(services) >= 15
    assert any(s["name"] == "model_training" for s in services)
    assert any(s["name"] == "performance_metrics" for s in services)
    assert "pipeline" in payload
    assert payload["pipeline"]["core_pipeline"][-1]["step"] == "trade_alerts"
