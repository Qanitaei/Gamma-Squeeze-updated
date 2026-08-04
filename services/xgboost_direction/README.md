# XGBoost Direction Prediction

| | |
|--|--|
| **Service** | `xgboost_direction` |
| **Port** | `8005` |
| **Phase** | **6 — Complete** |

## Responsibility

Multi-horizon (1/3/5/10/20 Day Return) forecasts with:

- **Classification** — down / flat / up
- **Regression** — expected forward return
- **Probability** — class probabilities

Training uses **TimeSeriesSplit CV**, **Optuna Bayesian optimization**, and **SHAP**.

See [docs/XGBOOST_DIRECTION.md](../../docs/XGBOOST_DIRECTION.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.xgboost_direction.app:app --host 0.0.0.0 --port 8005
```

```bash
curl -s http://127.0.0.1:8005/v1/meta
curl -s -X POST http://127.0.0.1:8005/v1/direction \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","train":true,"n_trials":15,"include_shap":true}'
```

Docker:

```bash
docker compose up -d xgboost_direction
curl -s http://127.0.0.1:8005/health
```
