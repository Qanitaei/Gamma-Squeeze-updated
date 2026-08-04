# Temporal Fusion Transformer Forecasting

| | |
|--|--|
| **Service** | `tft_forecasting` |
| **Port** | `8006` |
| **Phase** | **7 — Complete** |

## Responsibility

Multi-target quantile forecasts (Future Price, IV, Net GEX, Dealer Hedge Requirement, Gamma Flip, Call/Put Wall, Expected Move) over **1/2/3/5/10 trading days**, with median, q10/q25/q75/q90, prediction intervals, and attention weights.

See [docs/TFT_FORECASTING.md](../../docs/TFT_FORECASTING.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.tft_forecasting.app:app --host 0.0.0.0 --port 8006
```

```bash
curl -s http://127.0.0.1:8006/v1/meta
curl -s -X POST http://127.0.0.1:8006/v1/forecast \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","train":true,"quantile_mode":"median_spread"}'
```

Docker:

```bash
docker compose up -d tft_forecasting
curl -s http://127.0.0.1:8006/health
```

## Independence

This service **must** start and answer `/health` without peer services.
Upstream data may be missing; domain handlers return degraded structured JSON.

| Direction | Peers |
|-----------|--------|
| **Upstream** | feature_engineering (optional) |
| **Downstream** | gamma_squeeze_engine |
