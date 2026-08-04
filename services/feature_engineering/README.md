# Feature Engineering

| | |
|--|--|
| **Service** | `feature_engineering` |
| **Port** | `8002` |
| **Phase** | 4 — Feature Engineering (complete) |

## Responsibility

Dealer Positioning + Options Metrics + Technical Indicators + Macro Features (Fed/CPI/curve/FX/vol/credit) from Alpaca options matrices, OHLCV, and Fred-Economic-data KV.

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.feature_engineering.app:app --host 0.0.0.0 --port 8002
```

Docker:

```bash
docker compose up -d feature_engineering
curl -s http://127.0.0.1:8002/health
```

## Ops endpoints

- `GET /health`
- `GET /ready`
- `GET /docs`
- `GET /openapi.json`
- `GET /metrics`

## Primary API

- `GET|POST /v1/features`
- `GET|POST /v1/options-metrics`
- `GET|POST /v1/technical-indicators`
- `GET|POST /v1/macro-features`

## Independence

This service **must** start and answer `/health` without peer services.
Upstream data may be missing; domain handlers return degraded structured JSON.

| Direction | Peers |
|-----------|--------|
| **Upstream** | data_collection (optional live) |
| **Downstream** | pattern_recognition, regime_hmm, xgboost_direction, tft_forecasting |

Inter-service calls (if any) go only through `services.common.http_client` + registry URLs.

## Layout

- `app.py` — FastAPI HTTP shell
- `service.py` — orchestration glue
- Domain logic lives in `src/gamma_squeeze/` (no HTTP)
