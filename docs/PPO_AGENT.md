# PPO Reinforcement Learning Agent (Phase 11)

Service: `ppo_agent` `:8009`  
Modules: `src/gamma_squeeze/rl/{state,actions,reward,env,ppo}.py`

**Library:** Stable-Baselines3 PPO (`MlpPolicy`) over a Gymnasium discrete-action env. Heuristic fallback when SB3/model missing.

## State space

| Group | Label |
|-------|-------|
| `dealer_position` | Dealer Position |
| `regime` | Regime |
| `forecast` | Forecast |
| `portfolio` | Portfolio |
| `risk` | Risk |
| `exposure` | Exposure |
| `iv` | IV |
| `greeks` | Greeks |
| `macro` | Macro |

Observation dim: `STATE_DIM` (41).

## Actions

| ID | Name |
|----|------|
| 0 | Long |
| 1 | Short |
| 2 | Calls |
| 3 | Puts |
| 4 | Debit Spread |
| 5 | Credit Spread |
| 6 | Iron Condor |
| 7 | Calendar |
| 8 | Butterfly |
| 9 | Covered Call |
| 10 | Cash |
| 11 | Increase Hedge |
| 12 | Reduce Hedge |

## Reward

| Component | Role |
|-----------|------|
| Sharpe | Encourage risk-adjusted return |
| Sortino | Downside-aware return |
| Drawdown | Penalize peak-to-trough |
| Transaction Cost | Penalize turnover / structure cost |
| Slippage | Penalize execution friction |
| Tail Risk | Penalize CVaR-like left tail |
| Portfolio Variance | Penalize return variance |

## API

```bash
curl http://127.0.0.1:8009/v1/meta

curl -X POST http://127.0.0.1:8009/v1/action \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","train":true,"timesteps":1024,"horizon":5}'

curl 'http://127.0.0.1:8009/v1/action/AAPL?deterministic=true'
```

Install deps: `pip install -r requirements-rl.txt`

## Phase 11 complete criteria

- SB3 PPO train + act path works
- Locked state groups, 13 actions, 7 reward components
- Service `/v1/meta` documents the contract
- Tests pass (including short SB3 train); live symbol action succeeds
