import json
from pathlib import Path

import numpy as np
import pytest

from gamma_squeeze.core.container import Container, get_container, reset_container
from gamma_squeeze.core.logging import JsonFormatter, get_logger, log_event, setup_logging
from gamma_squeeze.core.metrics_collector import MetricsCollector
from gamma_squeeze.core.model_registry import (
    CallablePredictor,
    clear_predictors,
    get_predictor,
    list_predictors,
    register_predictor,
)
from gamma_squeeze.core.ports import PredictorPort
from gamma_squeeze.core.reproducibility import seed_everything, sorted_feature_matrix, stable_hash
from gamma_squeeze.core.settings import (
    clear_settings_cache,
    load_platform_settings,
    load_yaml_file,
    resolve_config_path,
)
from gamma_squeeze.core.standards import CODING_STANDARDS, PRODUCT_NORTH_STAR, standards_meta


@pytest.mark.unit
def test_locked_standards_catalog():
    assert len(CODING_STANDARDS) == 9
    assert "dependency injection" in CODING_STANDARDS[0].lower()
    assert "Strong typing" in CODING_STANDARDS[1]
    assert "unit and integration tests" in CODING_STANDARDS[2].lower()
    assert "Asynchronous" in CODING_STANDARDS[3]
    assert "Modular" in CODING_STANDARDS[4]
    assert "YAML" in CODING_STANDARDS[5]
    assert "Structured logging" in CODING_STANDARDS[6]
    assert "documentation" in CODING_STANDARDS[7].lower()
    assert "reproducible" in CODING_STANDARDS[8].lower()
    meta = standards_meta()
    assert meta["phase"] == 19
    assert meta["n_standards"] == 9
    assert "gamma squeezes" in PRODUCT_NORTH_STAR.lower()
    assert "explainable" in PRODUCT_NORTH_STAR.lower()
    import gamma_squeeze

    typed = Path(gamma_squeeze.__file__).resolve().parent / "py.typed"
    assert typed.is_file()


@pytest.mark.unit
def test_yaml_and_env_settings():
    clear_settings_cache()
    path = resolve_config_path()
    assert path.name == "platform.yaml"
    data = load_yaml_file(path)
    assert "platform" in data
    assert "models" in data
    s = load_platform_settings()
    assert s.python_requires == ">=3.12"
    assert s.global_seed == 42
    assert s.backend("squeeze") == "composite"
    assert s.log_json is True


@pytest.mark.unit
def test_dependency_injection_container():
    reset_container()
    c = get_container()
    assert c.has("settings")
    assert c.resolve("settings").name == "gamma-squeeze-platform"
    assert c.resolve("metrics").enabled in (True, False)
    c.register("custom", 123)
    assert c.get("custom") == 123
    local = Container()
    local.factory("x", lambda: {"ok": True})
    assert local.get("x")["ok"] is True
    reset_container()


@pytest.mark.unit
def test_structured_logging_json():
    setup_logging(level="INFO", json_logs=True)
    logger = get_logger("test_standards")
    log_event(logger, "hello", service="unit", symbol="AAPL", fold=1)
    fmt = JsonFormatter()
    record = logger.makeRecord("test_standards", 20, __file__, 1, "hello", (), None)
    record.service = "unit"
    record.extra_fields = {"fold": 1}
    line = fmt.format(record)
    payload = json.loads(line)
    assert payload["msg"] == "hello"
    assert payload["service"] == "unit"
    assert payload["fold"] == 1


@pytest.mark.unit
def test_metrics_collector():
    m = MetricsCollector(namespace="test", enabled=True)
    m.incr("calls", 2)
    m.gauge("auc", 0.7)
    with m.timer("fit"):
        _ = sum(range(1000))
    snap = m.snapshot()
    assert snap["counters"]["test.calls"] == 2
    assert snap["gauges"]["test.auc"] == 0.7
    assert snap["timings"]["test.fit"]["count"] == 1


@pytest.mark.unit
def test_reproducibility_and_deterministic_preprocess():
    s1 = seed_everything(123)
    a = np.random.rand(4)
    s2 = seed_everything(123)
    b = np.random.rand(4)
    assert s1 == s2 == 123
    assert np.allclose(a, b)
    h1 = stable_hash(np.array([1.0, 2.0, 3.0]))
    h2 = stable_hash(np.array([1.0, 2.0, 3.0]))
    assert h1 == h2
    cols, mat = sorted_feature_matrix(["z", "a", "m"], np.array([[1, 2, 3]], dtype=float))
    assert cols == ["a", "m", "z"]
    assert list(mat[0]) == [2.0, 3.0, 1.0]


@pytest.mark.unit
def test_modular_predictor_registry():
    clear_predictors()

    def _factory():
        return CallablePredictor("stub", lambda symbol, **kw: {"symbol": symbol, "ok": True})

    register_predictor("squeeze", _factory)
    assert "squeeze" in list_predictors()
    pred = get_predictor("squeeze")
    assert isinstance(pred, PredictorPort)
    assert pred.predict("MSFT")["ok"] is True
    # swap
    register_predictor(
        "squeeze",
        lambda: CallablePredictor("other", lambda symbol, **kw: {"symbol": symbol, "v": 2}),
    )
    assert get_predictor("squeeze").predict("MSFT")["v"] == 2
    clear_predictors()


@pytest.mark.integration
def test_service_async_health_and_metrics():
    from fastapi.testclient import TestClient

    from services.orchestrator.app import app

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    metrics = client.get("/metrics").json()
    assert "counters" in metrics or metrics.get("enabled") is False
    # async fan-out route exists
    resp = client.get("/v1/services/health/async")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "async"
    standards = client.get("/v1/standards").json()
    assert standards["phase"] == 19
    assert len(standards["standards"]) == 9
    assert "implementations" in standards
