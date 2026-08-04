# Model Training (Phase 16)

Service: `model_training` `:8014`  
Modules: `src/gamma_squeeze/training/{pipeline,validation,hyperopt,drift,versioning,auto_retrain}.py`

## Implement

| Capability | How |
|------------|-----|
| **Walk Forward Validation** | Expanding/rolling windows (`walk_forward_splits`) |
| **Time Series Split** | `sklearn.model_selection.TimeSeriesSplit` |
| **Bayesian Hyperparameter Search** | Optuna **TPESampler** |
| **Early Stopping** | Staged-predict patience + Optuna `MedianPruner` + plateau stop |
| **MLflow Tracking** | `stack.mlflow_tracking` (`MLFLOW_TRACKING_URI`) |
| **Version Control** | `models/versions/{name}/{version}` + registry activate/rollback |
| **Automatic Retraining** | Drift policy → fit → promote if AUC holds |
| **Drift Detection** | Combined report |
| **Feature Drift** | PSI per feature |
| **Prediction Drift** | KS + mean score shift |
| **Concept Drift** | AUC drop + label-rate shift |

## API

```bash
curl -s http://127.0.0.1:8014/v1/meta

curl -s -X POST http://127.0.0.1:8014/v1/train \
  -H 'content-type: application/json' \
  -d '{"symbols":["AAPL"],"n_trials":12,"validation_method":"walk_forward","auto_retrain":true}'

curl -s -X POST http://127.0.0.1:8014/v1/drift \
  -H 'content-type: application/json' \
  -d '{"symbols":["AAPL"]}'

curl -s 'http://127.0.0.1:8014/v1/versions?name=squeeze_probability'
```

## CLI

```bash
pip install -r requirements-train.txt
python scripts/run_model_training.py --symbols AAPL,MSFT --n-trials 15 --validation walk_forward
```

## Phase 16 complete criteria

- All eleven capabilities documented in `/v1/meta`
- Walk-forward and TSS validation paths work
- Optuna Bayesian search + early stopping
- MLflow logging when available
- Versioned artifacts + rollback
- Feature / prediction / concept drift drive auto-retrain
- Tests pass; live symbol train succeeds
