"""Phase 1 — System Architecture verification (no live Alpaca/SSD required)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.common.pipeline import (
    CORE_PIPELINE_STEPS,
    LOCKED_PORTS,
    PIPELINE_LABELS,
    PIPELINE_STEPS,
)
from services.common.registry import SERVICE_REGISTRY
from services.orchestrator.service import pipeline_contract, run_pipeline

ROOT = Path(__file__).resolve().parents[1]

# Locked core pipeline labels (must match ARCHITECTURE.md)
EXPECTED_CORE_ORDER = (
    "data_collection",
    "feature_engineering",
    "pattern_recognition",
    "regime_hmm",
    "xgboost_direction",
    "tft_forecasting",
    "dealer_hedging",
    "gamma_squeeze_engine",
    "ppo_agent",
    "portfolio_hedging",
    "dashboard",
    "trade_alerts",
)

SERVICE_APP_MODULES = {
    name: f"services.{name}.app" for name in LOCKED_PORTS
}


def test_core_pipeline_order_matches_architecture():
    assert CORE_PIPELINE_STEPS == EXPECTED_CORE_ORDER
    assert CORE_PIPELINE_STEPS[0] == "data_collection"
    assert CORE_PIPELINE_STEPS[-1] == "trade_alerts"
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "Data Collection (8001)" in text
    assert "Trade Alert API (8012)" in text
    # Core ends at Trade Alert; companions are separate
    assert "CORE_PIPELINE_STEPS" in text or "Trade Alert API" in text


def test_registry_ports_match_locked_map():
    assert set(SERVICE_REGISTRY) == set(LOCKED_PORTS)
    for name, port in LOCKED_PORTS.items():
        assert SERVICE_REGISTRY[name]["port"] == port
        assert SERVICE_REGISTRY[name]["name"] == name


def test_pipeline_steps_include_core_then_companions():
    assert PIPELINE_STEPS[: len(CORE_PIPELINE_STEPS)] == CORE_PIPELINE_STEPS
    for step in CORE_PIPELINE_STEPS:
        assert step in PIPELINE_LABELS
        assert step in SERVICE_REGISTRY


@pytest.mark.parametrize("name,module_name", sorted(SERVICE_APP_MODULES.items()))
def test_service_app_imports_and_health_ready(name: str, module_name: str):
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "app")
    client = TestClient(mod.app)

    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body.get("status") == "ok"
    assert body.get("service")

    ready = client.get("/ready")
    assert ready.status_code == 200
    ready_body = ready.json()
    assert "ready" in ready_body

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "openapi" in openapi.json()


def test_service_readme_contracts_exist():
    for name in LOCKED_PORTS:
        readme = ROOT / "services" / name / "README.md"
        assert readme.is_file(), f"missing {readme}"
        text = readme.read_text(encoding="utf-8")
        assert f"`{LOCKED_PORTS[name]}`" in text or str(LOCKED_PORTS[name]) in text
        assert "/health" in text
        assert "uvicorn" in text


def test_dockerfile_and_compose_exist():
    assert (ROOT / "Dockerfile").is_file()
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for name, port in LOCKED_PORTS.items():
        assert f"{name}:" in compose or name.replace("_", "-") in compose
        assert str(port) in compose
    assert "/health" in compose


def test_pipeline_contract_endpoint():
    from services.orchestrator.app import app

    client = TestClient(app)
    resp = client.get("/v1/pipeline/contract")
    assert resp.status_code == 200
    body = resp.json()
    steps = [s["step"] for s in body["core_pipeline"]]
    assert steps == list(CORE_PIPELINE_STEPS)
    assert "http" in body["modes"]
    assert "in_process" in body["modes"]


def test_run_pipeline_http_mode_records_per_stage_status():
    """HTTP mode with no peers: every stage errors, DAG still returns full status."""
    data = run_pipeline(
        "AAPL",
        mode="http",
        include_companions=False,
        http_timeout=0.5,
    )
    assert data["mode"] == "http"
    assert data["core_pipeline"] == list(CORE_PIPELINE_STEPS)
    assert len(data["stages"]) == len(CORE_PIPELINE_STEPS)
    assert data["ok"] is False
    assert data["core_ok"] is False
    for stage in data["stages"]:
        assert stage["status"] == "error"
        assert stage["ok"] is False
        assert stage["error"]
    # Stage failure must not abort — all stages present
    assert [s["step"] for s in data["stages"]] == list(CORE_PIPELINE_STEPS)


def test_pipeline_contract_helper_matches_constant():
    contract = pipeline_contract()
    assert [s["step"] for s in contract["core_pipeline"]] == list(CORE_PIPELINE_STEPS)
