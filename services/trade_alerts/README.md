# Alert Engine

| | |
|--|--|
| **Service** | `trade_alerts` |
| **Port** | `8012` |
| **Phase** | **15 — Complete** |

## Responsibility

Notify when squeeze probability exceeds threshold, dealer flip, gamma ramp, call/put wall breaks, IV expansion, large dealer hedge, pattern breakout, or earnings risk.

Channels: **WebSocket · Email · SMS · Discord · Slack · Webhook**.

See [docs/ALERT_ENGINE.md](../../docs/ALERT_ENGINE.md).

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.trade_alerts.app:app --host 0.0.0.0 --port 8012
```

```bash
curl -s http://127.0.0.1:8012/v1/meta
curl -s 'http://127.0.0.1:8012/v1/alerts/AAPL?min_probability=0.55'
```

Docker:

```bash
docker compose up -d trade_alerts
curl -s http://127.0.0.1:8012/health
```

## Independence

| Direction | Peers |
|-----------|--------|
| **Upstream** | gamma_squeeze_engine, dealer_hedging, patterns, calendar |
| **Downstream** | External channels (email/SMS/Discord/Slack/webhook) |
