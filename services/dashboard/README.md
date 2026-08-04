# Visualization Dashboard

| | |
|--|--|
| **Service** | `dashboard` |
| **Port** | `8011` |
| **Phase** | System Architecture (independent API) |

## Responsibility

Institutional JSON panels and heatmaps (no HTML desk in this service).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.dashboard.app:app --host 0.0.0.0 --port 8011
```

Docker:

```bash
docker compose up -d dashboard
curl -s http://127.0.0.1:8011/health
```

## Ops endpoints

- `GET /health`
- `GET /ready`
- `GET /docs`
- `GET /openapi.json`
- `GET /metrics`

## Primary API

- `GET /v1/meta`
- `GET|POST /v1/dashboard/{symbol}`

## Independence

This service **must** start and answer `/health` without peer services.
Upstream data may be missing; domain handlers return degraded structured JSON.

| Direction | Peers |
|-----------|--------|
| **Upstream** | gamma_squeeze_engine / explainability (optional) |
| **Downstream** | trade_alerts (display) |

Inter-service calls (if any) go only through `services.common.http_client` + registry URLs.

## Layout

- `app.py` — FastAPI HTTP shell
- `service.py` — orchestration glue
- Domain logic lives in `src/gamma_squeeze/` (no HTTP)
