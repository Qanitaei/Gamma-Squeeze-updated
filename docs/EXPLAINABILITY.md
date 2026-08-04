# Explainability Engine (Phase 13)

Service: `explainability` `:8013`  
Modules: `src/gamma_squeeze/explain/{engine,attributions}.py`

**Rule:** Every prediction must explain WHY it was produced.

## Provide

| Component | Payload key |
|-----------|-------------|
| **SHAP** | `shap` — TreeExplainer when available, else linear attribution |
| **Attention Maps** | `attention_maps` — temporal + variable (TFT when loaded) |
| **Feature Importance** | `feature_importance` — model native or \|SHAP\| normalized |
| **Counterfactual Analysis** | `counterfactuals` — feature flips that move P |
| **Partial Dependence** | `partial_dependence` — ICE curves on instance |
| **Decision Trace** | `decision_trace` — staged inputs → inference → attribution → decision |
| **Model Confidence** | `model_confidence` — score + High/Medium/Low/Very Low |

Narrative field: `why` (human-readable summary).

## API

```bash
curl http://127.0.0.1:8013/v1/meta

curl -X POST http://127.0.0.1:8013/v1/explain \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","include_forecast":true,"use_composite":true}'

curl 'http://127.0.0.1:8013/v1/explain/AAPL?use_composite=true'
```

Install: `pip install -r requirements-explain.txt` (includes SHAP).

## Phase 13 complete criteria

- All seven components returned on each explain call
- Non-empty `why` narrative always present
- Ensemble forecasts also attach explainability in `meta`
- Tests pass; live symbol explain succeeds
