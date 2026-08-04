# Performance Metrics

| | |
|--|--|
| **Service** | `performance_metrics` |
| **Port** | `8015` |
| **Phase** | **17 — Complete** |

## Responsibility

Track Accuracy → Expected Shortfall: classification, regression, trading ratios, and tail risk.

See [docs/PERFORMANCE_METRICS.md](../../docs/PERFORMANCE_METRICS.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.performance_metrics.app:app --host 0.0.0.0 --port 8015
```

```bash
curl -s http://127.0.0.1:8015/v1/meta
curl -s 'http://127.0.0.1:8015/v1/track/AAPL'
```

## Independence

| Direction | Peers |
|-----------|--------|
| **Upstream** | features, labels, model_training |
| **Downstream** | dashboard, alerts |
