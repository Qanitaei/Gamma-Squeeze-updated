"""Phase 2 — Programming Stack verification (no GPU / live infra required)."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from gamma_squeeze.stack.inventory import (
    CATEGORIES,
    LOCKED_STACK,
    PYTHON_REQUIRES,
    locked_stack,
    probe_stack,
)
from gamma_squeeze.stack.settings import get_stack_settings

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NAMES = {
    "language": ["Python"],
    "frameworks": [
        "PyTorch",
        "Lightning",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Stable-Baselines3",
        "Ray RLlib",
        "Darts",
        "PyTorch Forecasting",
        "Scikit-Learn",
        "SHAP",
        "Optuna",
        "MLflow",
    ],
    "backend": [
        "FastAPI",
        "Redis",
        "PostgreSQL",
        "DuckDB",
        "Apache Arrow",
        "Polars",
    ],
    "streaming": ["Kafka", "WebSockets"],
    "deployment": ["Docker", "Kubernetes"],
    "visualization": [
        "React",
        "Next.js",
        "Plotly",
        "TradingView Lightweight Charts",
        "ECharts",
    ],
}


def test_locked_stack_matches_specified_inventory():
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for item in LOCKED_STACK:
        by_cat[item["category"]].append(item["name"])
    for cat, names in EXPECTED_NAMES.items():
        assert by_cat[cat] == names, f"{cat}: {by_cat[cat]} != {names}"


def test_programming_stack_doc_lists_locked_components():
    text = (ROOT / "docs" / "PROGRAMMING_STACK.md").read_text(encoding="utf-8")
    assert "Python" in text and "3.12" in text
    for names in EXPECTED_NAMES.values():
        for name in names:
            assert name in text, f"missing {name} in PROGRAMMING_STACK.md"


def test_platform_yaml_stack_section():
    cfg = yaml.safe_load((ROOT / "config" / "platform.yaml").read_text(encoding="utf-8"))
    assert cfg["platform"]["python_requires"] == ">=3.12"
    assert cfg["platform"]["programming_stack_phase"] == 2
    stack = cfg["stack"]
    assert "PyTorch" in stack["frameworks"]
    assert "FastAPI" in stack["backend"]
    assert "Kafka" in stack["streaming"]
    assert "Docker" in stack["deployment"]
    assert "Next.js" in stack["visualization"]


def test_requirements_and_pyproject_cover_stack():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in pyproject
    assert "torch" in pyproject
    assert "stable-baselines3" in pyproject
    assert "darts" in pyproject
    assert "shap" in pyproject
    assert "optuna" in pyproject
    assert "mlflow" in pyproject
    assert "confluent-kafka" in pyproject
    assert "plotly" in pyproject

    assert (ROOT / "requirements-ml.txt").is_file()
    assert (ROOT / "requirements-rl.txt").is_file()
    assert (ROOT / "requirements-forecast.txt").is_file()
    assert (ROOT / "requirements-explain.txt").is_file()
    assert (ROOT / "requirements-train.txt").is_file()
    assert (ROOT / "requirements-streaming.txt").is_file()
    assert (ROOT / "requirements-viz.txt").is_file()

    core = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for pkg in ("fastapi", "redis", "duckdb", "pyarrow", "polars", "websockets"):
        assert pkg in core


def test_web_dashboard_package_has_viz_stack():
    pkg = (ROOT / "apps" / "web-dashboard" / "package.json").read_text(encoding="utf-8")
    for dep in (
        '"react"',
        '"next"',
        '"plotly.js"',
        '"lightweight-charts"',
        '"echarts"',
    ):
        assert dep in pkg


def test_deploy_artifacts_exist():
    assert (ROOT / "Dockerfile").is_file()
    assert (ROOT / "docker-compose.yml").is_file()
    assert (ROOT / "docker-compose.infra.yml").is_file()
    k8s = ROOT / "deploy" / "k8s"
    assert (k8s / "namespace.yaml").is_file()
    assert (k8s / "configmap.yaml").is_file()
    assert (k8s / "microservices.yaml").is_file()
    assert (k8s / "orchestrator.yaml").is_file()
    assert (k8s / "README.md").is_file()


def test_locked_stack_and_probe_helpers():
    inv = locked_stack()
    assert inv["phase"] == 2
    assert inv["python_requires"] == PYTHON_REQUIRES
    assert inv["python_ok"] is True
    probed = probe_stack()
    assert probed["core_ok"] is True
    assert isinstance(probed["probes"], list)
    assert len(probed["probes"]) == len(LOCKED_STACK)


def test_stack_settings_phase():
    get_stack_settings.cache_clear()
    s = get_stack_settings()
    assert s.python_requires == ">=3.12"
    assert s.phase == 2
    public = s.to_public_dict()
    assert "database_url" in public


def test_orchestrator_stack_endpoints():
    from services.orchestrator.app import app

    client = TestClient(app)
    inv = client.get("/v1/stack/inventory")
    assert inv.status_code == 200
    body = inv.json()
    assert body["phase"] == 2
    assert [c["name"] for c in body["categories"]["frameworks"]] == EXPECTED_NAMES[
        "frameworks"
    ]

    probed = client.get("/v1/stack/inventory?probe=true")
    assert probed.status_code == 200
    assert probed.json()["core_ok"] is True

    health = client.get("/v1/stack/health")
    assert health.status_code == 200
    h = health.json()
    assert h["phase"] == 2
    assert h["python_ok"] is True
    assert "inventory_summary" in h
