# Technical Pattern Recognition (Phase 8)

Service: `pattern_recognition` `:8003`  
Modules: `src/gamma_squeeze/patterns/recognition.py`, `hybrid_encoder.py`

## Hybrid detector

CNN (local structure) + LSTM/GRU (sequence) + Transformer (temporal attention) fused with classic geometric / candlestick rules.

Backend constant: `hybrid-cnn-lstm-transformer`.

## Recognize (chart patterns)

| Pattern |
|---------|
| Bull Flag |
| Bear Flag |
| Cup Handle |
| Ascending Triangle |
| Descending Triangle |
| Pennant |
| Double Bottom |
| Double Top |
| Head Shoulders |
| Inverse Head Shoulders |
| Rectangle |
| Wedge |
| Channel |
| Gap |
| Island Reversal |
| Volatility Squeeze |
| NR7 |
| Inside Bar |

## Candlestick library

Doji · Hammer · Shooting Star · Bullish/Bearish Engulfing · Morning/Evening Star · Three White Soldiers · Three Black Crows · Harami · Piercing Line · Dark Cloud Cover

## Return fields

| Field | Description |
|-------|-------------|
| **Pattern** | Detected name |
| **Confidence** | Fused rule + hybrid score `[0,1]` |
| **Target Projection** | `{price, pct, direction}` |
| **Probability** | Continuation / completion probability `[0,1]` |

## API

```bash
curl http://127.0.0.1:8003/v1/meta
curl http://127.0.0.1:8003/v1/catalog

curl -X POST http://127.0.0.1:8003/v1/patterns \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","lookback":80,"min_confidence":0.40,"include_candles":true}'

curl 'http://127.0.0.1:8003/v1/patterns/AAPL?lookback=80'
```

## Phase 8 complete criteria

- CNN/LSTM/Transformer hybrid scoring active
- Full chart catalog above recognized
- Candlestick library wired
- Each hit returns pattern, confidence, target projection, probability
- Tests pass; live symbol detect succeeds
