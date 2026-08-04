# Alert Engine (Phase 15)

Service: `trade_alerts` `:8012`  
Modules: `src/gamma_squeeze/alerts/{engine,channels}.py`

Data plane: Alpaca (options/dealer). No Schwab.

## Notify when

| Trigger | Default threshold |
|---------|-------------------|
| **Gamma Squeeze Probability** | `min_probability` 0.55 (configurable) |
| **Dealer Flip** | within `dealer_flip_distance_pct` 1.5% of flip |
| **Gamma Ramp** | `\|gamma_ramp\| >= gamma_ramp_abs` |
| **Call Wall Break** | spot ≥ call wall (− buffer) |
| **Put Wall Break** | spot ≤ put wall (+ buffer) |
| **Large IV Expansion** | ATM IV vs RVol ratio / abs |
| **Large Dealer Hedge** | `\|hedge shares\| >= dealer_hedge_shares` |
| **Pattern Breakout** | breakout catalog + confidence |
| **Earnings Risk** | calendar / earnings within look-ahead |

All thresholds are configurable via request body / query.

## Support channels

| Channel | Config |
|---------|--------|
| **WebSocket** | `/v1/ws/alerts?symbol=AAPL` (always on) |
| **Email** | `ALERT_SMTP_*`, `ALERT_EMAIL_TO` |
| **SMS** | Twilio `ALERT_TWILIO_*`, `ALERT_SMS_TO` |
| **Discord** | `ALERT_DISCORD_WEBHOOK_URL` |
| **Slack** | `ALERT_SLACK_WEBHOOK_URL` |
| **Webhook** | `ALERT_WEBHOOK_URL` |

## API

```bash
curl http://127.0.0.1:8012/v1/meta

curl -X POST http://127.0.0.1:8012/v1/alerts \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","min_probability":0.55,"notify":false}'

curl -X POST http://127.0.0.1:8012/v1/notify \
  -H 'content-type: application/json' \
  -d '{"symbol":"AAPL","min_probability":0.6,"channels":["Discord","Webhook"]}'
```

WebSocket: `ws://127.0.0.1:8012/v1/ws/alerts?symbol=AAPL&interval_s=15`

## Phase 15 complete criteria

- All nine triggers evaluated with configurable thresholds
- Six delivery channels supported (WS always; others when configured)
- Service `/v1/meta` documents triggers/channels/thresholds
- Tests pass; live symbol evaluate succeeds
