# Reinforcement Learning Trading Agent (PPO)

| | |
|--|--|
| **Service** | `ppo_agent` |
| **Port** | `8009` |
| **Phase** | **11 — Complete** |

## Responsibility

Stable-Baselines3 PPO policy over dealer/regime/forecast/portfolio/risk/exposure/IV/greeks/macro state.

Actions: Long · Short · Calls · Puts · Debit/Credit Spread · Iron Condor · Calendar · Butterfly · Covered Call · Cash · Increase/Reduce Hedge.

Reward: Sharpe · Sortino · Drawdown · Transaction Cost · Slippage · Tail Risk · Portfolio Variance.

See [docs/PPO_AGENT.md](../../docs/PPO_AGENT.md).

## Run independently

```bash
cd gamma-squeeze-platform
pip install -r requirements-rl.txt
PYTHONPATH=.:src uvicorn services.ppo_agent.app:app --host 0.0.0.0 --port 8009
```

```bash
curl -s http://127.0.0.1:8009/v1/meta
curl -s -X POST http://127.0.0.1:8009/v1/train \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","timesteps":1024}'
```

Docker:

```bash
docker compose up -d ppo_agent
curl -s http://127.0.0.1:8009/health
```

## Independence

| Direction | Peers |
|-----------|--------|
| **Upstream** | features, gamma_squeeze_engine (optional forecast) |
| **Downstream** | portfolio_hedging, alerts |
