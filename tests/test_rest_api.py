from fastapi.testclient import TestClient

from services.orchestrator.rest_api import REST_ENDPOINTS


def test_rest_catalog():
    assert "/options" in REST_ENDPOINTS
    assert "/backtest" in REST_ENDPOINTS
    assert "/train" in REST_ENDPOINTS
    assert "/retrain" in REST_ENDPOINTS
    assert "/health" in REST_ENDPOINTS
    assert "/docs" in REST_ENDPOINTS
    assert len(REST_ENDPOINTS) == 17


def test_health_and_api_catalog():
    from services.orchestrator.app import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    catalog = client.get("/api").json()
    assert catalog["phase"] == 18
    assert "/prediction" in catalog["implement"]
    assert len(catalog["implement"]) == 17
    assert client.get("/endpoints").json()["docs"] == "/docs"
    assert client.get("/docs").status_code == 200
    # Bare resource roots return usage help
    for root in (
        "/options",
        "/features",
        "/gex",
        "/gamma",
        "/regime",
        "/prediction",
        "/tft",
        "/xgboost",
        "/ppo",
        "/hedging",
        "/dashboard",
        "/alerts",
    ):
        help_body = client.get(root).json()
        assert help_body["path"] == root
        assert "GET" in help_body["methods"]
    # OpenAPI documents the REST paths
    schema = client.get("/openapi.json").json()
    paths = schema.get("paths") or {}
    for p in (
        "/options",
        "/options/{symbol}",
        "/features/{symbol}",
        "/gex/{symbol}",
        "/gamma/{symbol}",
        "/regime/{symbol}",
        "/prediction/{symbol}",
        "/tft/{symbol}",
        "/xgboost/{symbol}",
        "/ppo/{symbol}",
        "/hedging/{symbol}",
        "/dashboard/{symbol}",
        "/alerts/{symbol}",
        "/backtest",
        "/train",
        "/retrain",
        "/health",
    ):
        assert p in paths, p


def test_symbol_routes_mocked(monkeypatch):
    from services.orchestrator import rest_api as api
    from services.orchestrator.app import app

    monkeypatch.setattr(api, "handle_options", lambda *a, **k: {"symbol": "AAPL", "success": True, "as_of": "2025-01-02"})
    monkeypatch.setattr(api, "handle_features", lambda *a, **k: {"symbol": "AAPL", "n_rows": 10, "as_of": "2025-01-02"})
    monkeypatch.setattr(api, "handle_gex", lambda *a, **k: {"symbol": "AAPL", "success": True, "gex": {"net_gex": -1.0}})
    monkeypatch.setattr(api, "handle_gamma", lambda *a, **k: {"symbol": "AAPL", "as_of": "2025-01-02", "probability": 0.6})
    monkeypatch.setattr(api, "handle_regime", lambda *a, **k: {"symbol": "AAPL", "current_state": "Neutral"})
    monkeypatch.setattr(api, "handle_prediction", lambda *a, **k: {"symbol": "AAPL", "success": True, "prediction": {}})
    monkeypatch.setattr(api, "handle_tft", lambda *a, **k: {"symbol": "AAPL", "forecast": {}})
    monkeypatch.setattr(api, "handle_xgboost", lambda *a, **k: {"symbol": "AAPL", "direction": "up"})
    monkeypatch.setattr(api, "handle_ppo", lambda *a, **k: {"symbol": "AAPL", "action": "Cash"})
    monkeypatch.setattr(api, "handle_hedging", lambda *a, **k: {"symbol": "AAPL", "success": True})
    monkeypatch.setattr(api, "handle_dashboard", lambda *a, **k: {"symbol": "AAPL", "panels": []})
    monkeypatch.setattr(api, "handle_alerts", lambda *a, **k: {"symbol": "AAPL", "alerts": []})

    client = TestClient(app)
    for path in (
        "/options/AAPL",
        "/features/AAPL",
        "/gex/AAPL",
        "/gamma/AAPL",
        "/regime/AAPL",
        "/prediction/AAPL",
        "/tft/AAPL",
        "/xgboost/AAPL",
        "/ppo/AAPL",
        "/hedging/AAPL",
        "/dashboard/AAPL",
        "/alerts/AAPL",
    ):
        resp = client.get(path)
        assert resp.status_code == 200, path
        body = resp.json()
        assert body["service"]
        assert body["symbol"] == "AAPL"


def test_train_retrain_backtest_mocked(monkeypatch):
    from services.orchestrator import rest_api as api
    from services.orchestrator.app import app

    monkeypatch.setattr(
        api,
        "handle_train",
        lambda req: {"symbols": ["AAPL"], "status": "ok", "metrics": {}},
    )
    monkeypatch.setattr(
        api,
        "handle_retrain",
        lambda req: {"symbols": ["AAPL"], "status": "ok", "metrics": {}},
    )
    monkeypatch.setattr(
        api,
        "handle_backtest",
        lambda req: {"symbols": ["AAPL"], "status": "ok", "validation": {}, "metrics": {}},
    )
    client = TestClient(app)
    assert client.post("/train", json={"symbols": ["AAPL"]}).json()["data"]["status"] == "ok"
    assert client.post("/retrain", json={"symbols": ["AAPL"]}).json()["data"]["status"] == "ok"
    assert client.post("/backtest", json={"symbols": ["AAPL"]}).json()["data"]["status"] == "ok"


def test_registry_includes_rest():
    from services.orchestrator.app import app

    client = TestClient(app)
    reg = client.get("/v1/registry").json()
    assert "/options" in reg["rest"]
    assert len(reg["services"]) >= 15
