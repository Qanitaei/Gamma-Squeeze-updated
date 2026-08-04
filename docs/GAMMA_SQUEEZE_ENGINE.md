# Gamma Squeeze Engine (Phase 10)

Service: `gamma_squeeze_engine` `:8008`  
Module: `src/gamma_squeeze/models/composite_squeeze.py`

Composite probability model blending upstream engines into scores and a final prognosis.

## Inputs

| Key | Label |
|-----|-------|
| `hmm` | Hidden Markov |
| `xgboost` | XGBoost |
| `tft` | Temporal Transformer |
| `dealer` | Dealer Simulation |
| `patterns` | Pattern Recognition |
| `macro` | Macro |
| `sentiment` | Sentiment |

## Calculate (scores)

| Key | Label |
|-----|-------|
| `gamma_squeeze_score` | Gamma Squeeze Score |
| `gamma_expansion_score` | Gamma Expansion Score |
| `dealer_flow_score` | Dealer Flow Score |
| `momentum_score` | Momentum Score |
| `liquidity_score` | Liquidity Score |
| `breakout_score` | Breakout Score |

## Final output

| Key | Label |
|-----|-------|
| `probability` | Probability |
| `magnitude` | Magnitude (% expected move) |
| `expected_duration` | Expected Duration (days) |
| `expected_start` | Expected Start (days ahead; 0 = imminent) |
| `confidence` | Confidence |
| `risk_rating` | Risk Rating (`Low` / `Medium` / `High` / `Extreme`) |

## Blend sketch

```
P = 0.28·P_ensemble
  + 0.22·squeeze + 0.14·expansion + 0.12·dealer_flow
  + 0.10·momentum + 0.08·breakout + 0.06·liquidity_mid
  + regime_boost
```

Confidence blends HMM/XGB agreement with pattern support. Risk rating combines probability, dealer exhaustion, magnitude, and confidence.

## API

```bash
curl http://127.0.0.1:8008/v1/meta

curl -X POST http://127.0.0.1:8008/v1/squeeze \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","persist":true,"composite":true}'

curl 'http://127.0.0.1:8008/v1/squeeze/AAPL?persist=false'
```

## Phase 10 complete criteria

- All seven inputs collected (degraded-safe on missing peers)
- Six scores in `[0,1]`
- Final output fields populated with risk rating enum
- Service API documents locked contract
- Tests pass; live symbol squeeze succeeds
