# Market Regime Detection (Hidden Markov Model)

| | |
|--|--|
| **Service** | `regime_hmm` |
| **Port** | `8004` |
| **Phase** | 5 — Hidden Markov Model |

## Responsibility

16-state HMM (Bull…Panic). Outputs: **current_state**, **transition_matrix**, **next_state_probability**, **confidence**.

## Run independently

```bash
cd gamma-squeeze-platform
PYTHONPATH=.:src uvicorn services.regime_hmm.app:app --host 0.0.0.0 --port 8004
```

Docker:

```bash
docker compose up -d regime_hmm
curl -s http://127.0.0.1:8004/health
```

## Ops endpoints

- `GET /health`
- `GET /ready`
- `GET /docs`
- `GET /openapi.json`
- `GET /metrics`

## Primary API

- `GET /v1/states`
- `GET|POST /v1/regime/{symbol}`

## Independence

This service **must** start and answer `/health` without peer services.
Upstream data may be missing; domain handlers return degraded structured JSON.

| Direction | Peers |
|-----------|--------|
| **Upstream** | feature_engineering (optional) |
| **Downstream** | gamma_squeeze_engine, ppo_agent |

Inter-service calls (if any) go only through `services.common.http_client` + registry URLs.

## Layout

- `app.py` — FastAPI HTTP shell
- `service.py` — orchestration glue
- Domain logic lives in `src/gamma_squeeze/` (no HTTP)
