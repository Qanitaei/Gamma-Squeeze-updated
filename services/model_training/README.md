# Model Training

| | |
|--|--|
| **Service** | `model_training` |
| **Port** | `8014` |
| **Phase** | **16 — Complete** |

## Responsibility

Walk Forward · Time Series Split · Bayesian HPO · Early Stopping · MLflow · Version Control · Automatic Retraining · Feature/Prediction/Concept Drift.

See [docs/MODEL_TRAINING.md](../../docs/MODEL_TRAINING.md).

## Run independently

```bash
cd gamma-squeeze-platform
pip install -r requirements-train.txt
PYTHONPATH=.:src uvicorn services.model_training.app:app --host 0.0.0.0 --port 8014
```

```bash
curl -s http://127.0.0.1:8014/v1/meta
curl -s -X POST http://127.0.0.1:8014/v1/train \
  -H 'content-type: application/json' \
  -d '{"symbols":["AAPL"],"n_trials":8,"force_retrain":true}'
```

## Independence

| Direction | Peers |
|-----------|--------|
| **Upstream** | feature_engineering / labels |
| **Downstream** | gamma_squeeze_engine, performance_metrics |
