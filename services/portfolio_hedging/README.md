# Portfolio Hedging Engine

| | |
|--|--|
| **Service** | `portfolio_hedging` |
| **Port** | `8010` |
| **Phase** | **12 — Complete** |

## Responsibility

Recommend Long Calls/Puts, Covered/Protective, Debit/Credit/Calendar Spreads, Iron Condor, Collar, Dynamic Delta/Gamma/Vega/Theta hedges.

Optimize for **Return · Risk · Capital Efficiency · Margin**.

See [docs/PORTFOLIO_HEDGING.md](../../docs/PORTFOLIO_HEDGING.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.portfolio_hedging.app:app --host 0.0.0.0 --port 8010
```

```bash
curl -s http://127.0.0.1:8010/v1/meta
curl -s 'http://127.0.0.1:8010/v1/hedges/AAPL?optimize_for=Capital%20Efficiency'
```

Docker:

```bash
docker compose up -d portfolio_hedging
curl -s http://127.0.0.1:8010/health
```

## Independence

| Direction | Peers |
|-----------|--------|
| **Upstream** | features, ensemble/composite forecast, Alpaca matrix |
| **Downstream** | dashboard, alerts |
