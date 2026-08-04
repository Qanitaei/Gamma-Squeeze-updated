# API Gateway / Orchestrator

| | |
|--|--|
| **Service** | `orchestrator` |
| **Port** | `8000` |
| **Phase** | **18 — Complete** (unified REST) |

## Responsibility

Unified REST (`/options` … `/retrain`, `/health`, `/docs`) + ordered pipeline DAG.

See [docs/API_ENDPOINTS.md](../../docs/API_ENDPOINTS.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.orchestrator.app:app --host 0.0.0.0 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/api
curl -s 'http://127.0.0.1:8000/features/AAPL'
```

## REST (Phase 18)

`/options` · `/features` · `/gex` · `/gamma` · `/regime` · `/prediction` · `/tft` · `/xgboost` · `/ppo` · `/hedging` · `/dashboard` · `/alerts` · `/backtest` · `/train` · `/retrain` · `/health` · `/docs`

Also: `GET /api`, `GET /endpoints`, `/v1/registry`, `/v1/standards`, `/v1/pipeline/run`.

## Independence

This service **must** start and answer `/health` without peer services.
Upstream data may be missing; domain handlers return degraded structured JSON.

| Direction | Peers |
|-----------|--------|
| **Upstream** | None (gateway) |
| **Downstream** | All pipeline peers via HTTP or in-process |
