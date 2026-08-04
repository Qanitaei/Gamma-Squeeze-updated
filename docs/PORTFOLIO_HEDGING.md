# Portfolio Hedging Engine (Phase 12)

Service: `portfolio_hedging` `:8010`  
Module: `src/gamma_squeeze/risk/portfolio_hedging_engine.py`

Data plane: Alpaca (SSD → API → KV).

## Recommend

| Structure |
|-----------|
| Long Calls |
| Long Puts |
| Covered Calls |
| Protective Puts |
| Debit Spreads |
| Credit Spreads |
| Calendar Spreads |
| Iron Condor |
| Collar |
| Dynamic Delta Hedge |
| Gamma Hedge |
| Vega Hedge |
| Theta Hedge |

Each structure is scored on suitability given dealer book, regime, and squeeze forecast (composite when available).

## Optimize

Primary objective (`optimize_for`):

| Objective | Meaning |
|-----------|---------|
| **Return** | Favor higher expected payoff under squeeze path |
| **Risk** | Favor protective / defined-risk overlays |
| **Capital Efficiency** | Favor spreads / condors / calendars |
| **Margin** | Favor lower margin burden |

Returns top-k weights plus expected scores across all four objectives.

## API

```bash
curl http://127.0.0.1:8010/v1/meta
curl http://127.0.0.1:8010/v1/structures

curl 'http://127.0.0.1:8010/v1/hedges/AAPL?optimize_for=Risk'

curl -X POST http://127.0.0.1:8010/v1/optimize \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","optimize_for":"Capital Efficiency","use_live_alpaca":true}'
```

## Phase 12 complete criteria

- All 13 structures recommended with Return/Risk/Capital Efficiency/Margin scores
- Optimizer switches ranking by primary objective
- Service API documents locked contract
- Tests pass; live symbol hedge plan succeeds
