# Technical Pattern Recognition

| | |
|--|--|
| **Service** | `pattern_recognition` |
| **Port** | `8003` |
| **Phase** | **8 — Complete** |

## Responsibility

CNN/LSTM/Transformer hybrid detection of chart patterns + candlestick library. Each hit returns **pattern**, **confidence**, **target projection**, and **probability**.

See [docs/PATTERN_RECOGNITION.md](../../docs/PATTERN_RECOGNITION.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.pattern_recognition.app:app --host 0.0.0.0 --port 8003
```

```bash
curl -s http://127.0.0.1:8003/v1/meta
curl -s 'http://127.0.0.1:8003/v1/patterns/AAPL?lookback=80'
```

Docker:

```bash
docker compose up -d pattern_recognition
curl -s http://127.0.0.1:8003/health
```

## Independence

This service **must** start and answer `/health` without peer services.

| Direction | Peers |
|-----------|--------|
| **Upstream** | OHLCV ingest (Alpaca) |
| **Downstream** | gamma_squeeze_engine, alerts |
