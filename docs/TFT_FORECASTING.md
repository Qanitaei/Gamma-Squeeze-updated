# Temporal Fusion Transformer (Phase 7)

Service: `tft_forecasting` `:8006`  
Module: `src/gamma_squeeze/deep/temporal_model.py`

## Forecast targets

| Key | Label |
|-----|-------|
| `future_price` | Future Price |
| `iv` | IV |
| `net_gex` | Net GEX |
| `dealer_hedge_requirement` | Dealer Hedge Requirement |
| `gamma_flip` | Gamma Flip |
| `call_wall` | Call Wall |
| `put_wall` | Put Wall |
| `expected_move` | Expected Move |

Constant: `TFT_TARGETS`.

## Forecast horizons (trading days)

| Horizon | Label |
|---------|-------|
| 1 | 1 Trading Day |
| 2 | 2 Trading Days |
| 3 | 3 Trading Days |
| 5 | 5 Trading Days |
| 10 | 10 Trading Days |

Constant: `TFT_HORIZONS = (1, 2, 3, 5, 10)`.

## Outputs (per target × horizon)

| Output | Description |
|--------|-------------|
| **Median Forecast** | `median` / `median_forecast` (q50) |
| **10%** | `q10` / `p10` |
| **25%** | `q25` / `p25` |
| **75%** | `q75` / `p75` |
| **90%** | `q90` / `p90` |
| **Prediction Interval** | `prediction_interval` — low/high (q10–q90), mid band (q25–q75) |
| **Attention Weights** | `attention_weights.temporal` + `.variables` |

Quantile modes: `full` (fit each quantile) or `median_spread` (fit median + residual σ).

Backend: TFT-style temporal/variable attention + sklearn quantile GBTs (PyTorch Forecasting optional when installed).

## API

```bash
curl http://127.0.0.1:8006/v1/meta
curl http://127.0.0.1:8006/v1/horizons

curl -X POST http://127.0.0.1:8006/v1/forecast \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","train":true,"lookback":32,"quantile_mode":"median_spread"}'

curl 'http://127.0.0.1:8006/v1/forecast/AAPL?train=false'
```

## Phase 7 complete criteria

- All eight forecast targets available
- Horizons 1/2/3/5/10 trading days
- Median + q10/q25/q75/q90 + prediction interval per cell
- Attention weights (temporal + variable) on each forecast
- Service API documents the locked contract
- Tests pass; live symbol train/infer succeeds
