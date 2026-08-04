# Performance Metrics (Phase 17)

Service: `performance_metrics` `:8015`  
Modules: `src/gamma_squeeze/metrics/{tracker,classification,regression,trading,risk}.py`

## Track

| Group | Metrics |
|-------|---------|
| **Classification** | Accuracy · Precision · Recall · F1 · ROC AUC · Brier Score · Log Loss |
| **Regression** | MAE · RMSE |
| **Trading** | Sharpe · Sortino · Calmar · Maximum Drawdown · Annual Return · Win Rate · Profit Factor · Average Trade · Average Hold Time |
| **Risk** | Tail Risk · Expected Shortfall |

Constant: `TRACKED_METRICS` (20 metrics).

## API

```bash
curl -s http://127.0.0.1:8015/v1/meta

curl -s 'http://127.0.0.1:8015/v1/track/AAPL'

curl -s -X POST http://127.0.0.1:8015/v1/compute \
  -H 'content-type: application/json' \
  -d '{"y_true":[0,1,1,0],"proba":[0.2,0.8,0.7,0.3],"returns":[0.01,-0.005,0.02]}'
```

Symbol track: loads features/labels, fits a holdout classifier, scores classification + magnitude MAE/RMSE + strategy trading/risk metrics.

## Phase 17 complete criteria

- All 20 catalog metrics available via `/v1/meta`
- Classification / regression / trading / risk blocks populated
- `/v1/track` and `/v1/compute` paths work
- Tests pass; live symbol track succeeds
