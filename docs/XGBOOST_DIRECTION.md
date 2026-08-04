# XGBoost Direction Model (Phase 6)

Service: `xgboost_direction` `:8005`  
Module: `src/gamma_squeeze/models/xgboost_direction.py`

## Predictions (per horizon)

| Horizon | Key | Label |
|---------|-----|-------|
| 1 Day Return | `1d` | `1 Day Return` |
| 3 Day Return | `3d` | `3 Day Return` |
| 5 Day Return | `5d` | `5 Day Return` |
| 10 Day Return | `10d` | `10 Day Return` |
| 20 Day Return | `20d` | `20 Day Return` |

Constant: `DIRECTION_HORIZONS = (1, 3, 5, 10, 20)`.

## Targets

| Target | Description |
|--------|-------------|
| **Classification** | `down` / `flat` / `up` vs threshold (default ±0.5%) |
| **Regression** | Expected forward return over the horizon |
| **Probability** | Softmax class probabilities (`multi:softprob` / booster equivalent) |

## Training

| Method | Implementation |
|--------|----------------|
| **Cross Validation** | `sklearn.model_selection.TimeSeriesSplit` |
| **Bayesian Optimization** | Optuna TPE over tree hyperparameters |
| **SHAP Explainability** | TreeExplainer (fallback: feature importances) |

Backend fallback chain: **XGBoost** → LightGBM → CatBoost → HistGradientBoosting.

## API

```bash
curl http://127.0.0.1:8005/v1/horizons
curl http://127.0.0.1:8005/v1/meta

curl -X POST http://127.0.0.1:8005/v1/direction \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","train":true,"n_trials":15,"cv_folds":5,"include_shap":true}'

# single horizon
curl 'http://127.0.0.1:8005/v1/direction/AAPL?horizon=5&train=false'
```

## Phase 6 complete criteria

- Horizons 1/3/5/10/20 trading days
- Each horizon emits classification, regression, probability
- Training path uses TimeSeriesSplit CV + Optuna Bayesian search + SHAP
- Service API documents horizons/targets/training methods
- Tests pass on synthetic panels; live symbol train/infer succeeds
