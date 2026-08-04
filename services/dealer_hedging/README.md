# Dealer Hedging Simulation Engine

| | |
|--|--|
| **Service** | `dealer_hedging` |
| **Port** | `8007` |
| **Phase** | **9 — Complete** |

## Responsibility

Simulate dealer hedging: share purchases/sales, gamma ramp, strike migration, delta hedging, dynamic gamma, inventory, liquidity, exhaustion, position flip.

Outputs: **Dealer Flow Map**, **Dealer Demand Curve**, **Expected Hedging Volume**.

See [docs/DEALER_HEDGING.md](../../docs/DEALER_HEDGING.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.dealer_hedging.app:app --host 0.0.0.0 --port 8007
```

```bash
curl -s http://127.0.0.1:8007/v1/meta
curl -s -X POST http://127.0.0.1:8007/v1/simulate \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","n_points":17}'
```

Docker:

```bash
docker compose up -d dealer_hedging
curl -s http://127.0.0.1:8007/health
```

## Independence

| Direction | Peers |
|-----------|--------|
| **Upstream** | Alpaca options matrix (SSD / API / KV) |
| **Downstream** | gamma_squeeze_engine, alerts |
