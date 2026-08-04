# Data Collection

| | |
|--|--|
| **Service** | `data_collection` |
| **Port** | `8001` |
| **Phase** | 3 — Data Sources (independent API) |

## Responsibility

Multi-source ingest: Historical Options Matrix (Alpaca, Schwab, IBKR, OPRA, Polygon, CBOE), Market Data, Macro, Structure (VIX/VVIX/MOVE/DXY/USDJPY/yields), Corporate, Calendar. See [docs/DATA_SOURCES.md](../../docs/DATA_SOURCES.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.data_collection.app:app --host 0.0.0.0 --port 8001
```

Docker:

```bash
docker compose up -d data_collection
curl -s http://127.0.0.1:8001/health
```

## Ops endpoints

- `GET /health`
- `GET /ready`
- `GET /docs`
- `GET /openapi.json`
- `GET /metrics`

## Primary API

- `POST /v1/collect`
- `GET /v1/collect/{symbol}`
- `GET /v1/adapters`
- `GET /v1/adapters/health`

## Independence

This service **must** start and answer `/health` without peer services.
Upstream data may be missing; domain handlers return degraded structured JSON.

| Direction | Peers |
|-----------|--------|
| **Upstream** | External: Alpaca / SSD / KV |
| **Downstream** | feature_engineering, dealer_hedging |

Inter-service calls (if any) go only through `services.common.http_client` + registry URLs.

## Layout

- `app.py` — FastAPI HTTP shell
- `service.py` — orchestration glue
- Domain logic lives in `src/gamma_squeeze/` (no HTTP)
