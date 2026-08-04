# API Endpoints (Phase 18)

Gateway: `orchestrator` `:8000`  
Module: `services/orchestrator/rest_api.py`

Unified REST surface (in-process fan-out to domain services). Also: FastAPI `/health`, `/docs`, `/openapi.json`.

## Implement

| Path | Methods | Backend |
|------|---------|---------|
| `/options` | GET, GET `/{symbol}`, POST | Options matrix collection + options metrics |
| `/features` | GET, GET `/{symbol}`, POST | Feature engineering panel |
| `/gex` | GET, GET `/{symbol}`, POST | GEX/DEX snapshot from Alpaca/SSD matrix |
| `/gamma` | GET, GET `/{symbol}`, POST | Composite gamma squeeze engine |
| `/regime` | GET, GET `/{symbol}`, POST | HMM regime |
| `/prediction` | GET, GET `/{symbol}`, POST | Squeeze forecast / prediction bundle |
| `/tft` | GET, GET `/{symbol}`, POST | Temporal Fusion Transformer |
| `/xgboost` | GET, GET `/{symbol}`, POST | XGBoost direction |
| `/ppo` | GET, GET `/{symbol}`, POST | PPO agent action |
| `/hedging` | GET, GET `/{symbol}`, POST | Dealer + portfolio hedging |
| `/dashboard` | GET, GET `/{symbol}`, POST | Institutional dashboard panels |
| `/alerts` | GET, GET `/{symbol}`, POST | Alert engine evaluation |
| `/backtest` | GET, POST | Walk-forward validation + performance metrics |
| `/train` | GET, POST | Model training (Bayesian HPO / MLflow / drift) |
| `/retrain` | GET, POST | Continuous retrain (`baseline` \| `rl` \| `deep` \| `training`) |
| `/health` | GET | Liveness |
| `/docs` | GET | OpenAPI Swagger UI |

Bare `GET /{resource}` (without symbol) returns usage/help JSON.  
Catalog: `GET /api` or `GET /endpoints` · legacy microservice routes remain under `/v1/*` on each service.

Constant: `REST_ENDPOINTS` (17 paths).

## Examples

```bash
PYTHONPATH=.:src uvicorn services.orchestrator.app:app --port 8000

curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/api | jq .
curl -s http://127.0.0.1:8000/options | jq .
curl -s 'http://127.0.0.1:8000/features/AAPL' | jq .
curl -s 'http://127.0.0.1:8000/gex/AAPL' | jq .
curl -s 'http://127.0.0.1:8000/prediction/AAPL' | jq .
curl -s -X POST http://127.0.0.1:8000/train \
  -H 'content-type: application/json' \
  -d '{"symbols":["AAPL"],"n_trials":10,"force_retrain":true}'
curl -s -X POST http://127.0.0.1:8000/backtest \
  -H 'content-type: application/json' \
  -d '{"symbols":["AAPL"],"n_splits":3}'
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Phase 18 complete criteria

- All 17 catalog paths registered on the gateway
- Symbol routes + job routes (`backtest` / `train` / `retrain`) return `ServiceEnvelope`
- `/health` and `/docs` live
- OpenAPI includes the REST surface; tests pass
