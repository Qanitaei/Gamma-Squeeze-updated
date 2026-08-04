# Explainability Engine

| | |
|--|--|
| **Service** | `explainability` |
| **Port** | `8013` |
| **Phase** | **13 — Complete** |

## Responsibility

Every prediction explains **WHY**: SHAP · Attention Maps · Feature Importance · Counterfactual Analysis · Partial Dependence · Decision Trace · Model Confidence.

See [docs/EXPLAINABILITY.md](../../docs/EXPLAINABILITY.md).

## Run independently

```bash
cd gamma-squeeze-platform
pip install -r requirements-explain.txt
PYTHONPATH=.:src uvicorn services.explainability.app:app --host 0.0.0.0 --port 8013
```

```bash
curl -s http://127.0.0.1:8013/v1/meta
curl -s 'http://127.0.0.1:8013/v1/explain/AAPL?use_composite=true'
```

Docker:

```bash
docker compose up -d explainability
curl -s http://127.0.0.1:8013/health
```

## Independence

| Direction | Peers |
|-----------|--------|
| **Upstream** | features, ensemble/composite, TFT (attention), probability models |
| **Downstream** | dashboard, alerts |
