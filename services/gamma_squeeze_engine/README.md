# Gamma Squeeze Probability Engine

| | |
|--|--|
| **Service** | `gamma_squeeze_engine` |
| **Port** | `8008` |
| **Phase** | **10 — Complete** |

## Responsibility

Composite probability model from HMM + XGBoost + TFT + Dealer Simulation + Patterns + Macro + Sentiment.

Calculates squeeze / expansion / dealer-flow / momentum / liquidity / breakout scores.  
Final output: **Probability · Magnitude · Expected Duration · Expected Start · Confidence · Risk Rating**.

See [docs/GAMMA_SQUEEZE_ENGINE.md](../../docs/GAMMA_SQUEEZE_ENGINE.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.gamma_squeeze_engine.app:app --host 0.0.0.0 --port 8008
```

```bash
curl -s http://127.0.0.1:8008/v1/meta
curl -s 'http://127.0.0.1:8008/v1/squeeze/AAPL?persist=false'
```

Docker:

```bash
docker compose up -d gamma_squeeze_engine
curl -s http://127.0.0.1:8008/health
```

## Independence

| Direction | Peers |
|-----------|--------|
| **Upstream** | features, HMM, XGB, TFT, dealer, patterns, macro |
| **Downstream** | PPO, portfolio hedging, dashboard, alerts |
