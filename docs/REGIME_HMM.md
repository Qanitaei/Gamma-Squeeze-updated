# Hidden Markov Model — Market Regimes (Phase 5)

Service: `regime_hmm` `:8004`  
Module: `src/gamma_squeeze/regime/hmm_regime.py`

## Hidden states (16)

| # | State |
|---|--------|
| 0 | Bull |
| 1 | Strong Bull |
| 2 | Neutral |
| 3 | Compression |
| 4 | Distribution |
| 5 | Accumulation |
| 6 | Positive Gamma |
| 7 | Negative Gamma |
| 8 | Dealer Neutral |
| 9 | High Volatility |
| 10 | Low Volatility |
| 11 | Gamma Expansion |
| 12 | Gamma Squeeze |
| 13 | Short Squeeze |
| 14 | Capitulation |
| 15 | Panic |

Constant: `REGIME_STATES` / `N_STATES = 16`.

## Method

1. Build observation vector from the feature store (returns, realized vol, GEX/dealer, PCR, technicals, VIX/macro).
2. Score each named state with rule-informed emission logits → softmax probabilities.
3. Estimate a sticky transition matrix from the hard-assigned state path (Laplace-smoothed).
4. Forward-filter posteriors; optional GaussianHMM blend when data-rich (`hmmlearn`).
5. Emit locked outputs below.

## Output (locked)

| Field | Meaning |
|-------|---------|
| `current_state` | MAP regime at latest as-of |
| `transition_matrix` | Labeled 16×16 row-stochastic matrix |
| `next_state_probability` | `posterior @ P` one-step ahead |
| `confidence` | Blend of top posterior mass + inverse entropy |

Also returned: `state_posterior`, `path`, `state_labels`, `n_obs`, `version`.

## API

```bash
curl http://127.0.0.1:8004/v1/states
curl http://127.0.0.1:8004/v1/meta

curl -X POST http://127.0.0.1:8004/v1/regime \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","train":true,"include_macro":true}'

curl 'http://127.0.0.1:8004/v1/regime/AAPL?train=true'
```

## Phase 5 complete criteria

- All 16 named hidden states registered
- Inference returns `current_state`, `transition_matrix`, `next_state_probability`, `confidence`
- Transition rows sum to 1; next-state probs sum to 1
- Tests pass without live broker (synthetic feature frames)
