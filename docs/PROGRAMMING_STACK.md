# Programming Stack (Phase 2)

Canonical, locked stack for the Gamma Squeeze Platform. Every later phase
(models, engines, dashboard) must use these components — not alternate frameworks.

**Phase status:** locked. See [ARCHITECTURE.md](../ARCHITECTURE.md) phased delivery.

## Language

| Component | Version | Install |
|-----------|---------|---------|
| **Python** | **3.12+** | Runtime / Docker `python:3.12-slim` / `requires-python = ">=3.12"` |

## Frameworks (ML / forecasting / RL / explain / train)

| Library | Role | Extra / requirements file |
|---------|------|---------------------------|
| **PyTorch** | Deep learning runtime | `ml` / `requirements-ml.txt` |
| **Lightning** | Training loops / multi-GPU | `ml` / `requirements-ml.txt` |
| **XGBoost** | Gradient-boosted direction / squeeze heads | `ml` / `requirements-ml.txt` |
| **LightGBM** | Fast tabular boosters | `ml` / `requirements-ml.txt` |
| **CatBoost** | Ordered-boosting tabular models | `ml` / `requirements-ml.txt` |
| **Stable-Baselines3** | PPO trading agents | `rl` / `requirements-rl.txt` |
| **Ray RLlib** | Distributed RL | `rl` / `requirements-rl.txt` |
| **Darts** | Time-series forecasting toolkit | `forecast` / `requirements-forecast.txt` |
| **PyTorch Forecasting** | TFT and related temporal models | `forecast` / `requirements-forecast.txt` |
| **Scikit-Learn** | Baselines, calibration, pipelines | **core** / `requirements.txt` |
| **SHAP** | Model explainability | `explain` / `requirements-explain.txt` |
| **Optuna** | Hyperparameter search | `train` / `requirements-train.txt` |
| **MLflow** | Experiment tracking / model registry | `train` / `requirements-train.txt` |

## Backend

| Component | Role | Install |
|-----------|------|---------|
| **FastAPI** | Independent microservice HTTP APIs | core |
| **Redis** | Cache, pub/sub alerts, rate limits | core client + `docker-compose.infra.yml` |
| **PostgreSQL** | Operational metadata, alerts, registry | core client + infra compose |
| **DuckDB** | Local analytical queries over SSD / Parquet | core |
| **Apache Arrow** | Columnar interchange (`pyarrow`) | core |
| **Polars** | High-performance feature frames | core |

## Streaming

| Component | Role | Install |
|-----------|------|---------|
| **Kafka** | Feature / forecast / alert event bus (Redpanda locally) | `streaming` / `requirements-streaming.txt` |
| **WebSockets** | Live dashboard + alert streaming | core (`websockets` + FastAPI WS) |

## Deployment

| Component | Role | Location |
|-----------|------|----------|
| **Docker** | Local + CI images (`python:3.12-slim`) | [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml), [`docker-compose.infra.yml`](../docker-compose.infra.yml) |
| **Kubernetes** | Production microservice deploy | [`deploy/k8s/`](../deploy/k8s/) |

## Visualization

| Component | Role | Location |
|-----------|------|----------|
| **React** | UI component model | [`apps/web-dashboard`](../apps/web-dashboard) |
| **Next.js** | Primary dashboard app | `apps/web-dashboard` (`next`) |
| **Plotly** | Research / interactive charts | `plotly.js` + `react-plotly.js`; Python optional `viz` extra |
| **TradingView Lightweight Charts** | Price / overlay charts | `lightweight-charts` |
| **ECharts** | Regime / distribution / hedge curves | `echarts` + `echarts-for-react` |

## Machine-readable inventory

Locked catalog + optional import probes:

- Module: `gamma_squeeze.stack.inventory`
- API: `GET /v1/stack/inventory` and `GET /v1/stack/health` on the orchestrator (`:8000`)

```bash
PYTHONPATH=.:src python -c "from gamma_squeeze.stack.inventory import locked_stack; print(locked_stack())"
```

## Install profiles

```bash
# lean API + backend baselines (default)
pip install -e .
# or
pip install -r requirements.txt

# full quantitative stack
pip install -e ".[ml,rl,forecast,explain,train,streaming,viz]"

# layered requirements
pip install -r requirements.txt
pip install -r requirements-ml.txt
pip install -r requirements-rl.txt
pip install -r requirements-forecast.txt
pip install -r requirements-explain.txt
pip install -r requirements-train.txt
pip install -r requirements-streaming.txt
pip install -r requirements-viz.txt
```

Visualization frontend:

```bash
cd apps/web-dashboard && npm install && npm run dev
```

## Infra + deploy

```bash
# data plane (Redis, PostgreSQL, Redpanda/Kafka, MLflow)
docker compose -f docker-compose.infra.yml up -d

# Python microservices (3.12 images)
docker compose up -d --build

# Kubernetes (namespace + all services)
kubectl apply -f deploy/k8s/
```

Defaults (override in `.env`):

| Variable | Default |
|----------|---------|
| `DATABASE_URL` | `postgresql://gamma:gamma@localhost:5432/gamma_squeeze` |
| `REDIS_URL` | `redis://localhost:6379/0` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` |
| `DUCKDB_PATH` | `{matrix_root}/analytics/gamma.duckdb` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` |

## Service dependency policy

- **Library layer** (`src/gamma_squeeze`): pure functions; optional heavy imports guarded via inventory probes.
- **Service layer** (`services/*`): thin FastAPI adapters.
- **Web layer** (`apps/web-dashboard`): Next.js consuming orchestrator + WebSocket alerts.
- **No circular service imports** — HTTP, Kafka topics, or shared schemas only.

## Phase 2 complete criteria

- This document lists exactly the locked Language / Frameworks / Backend / Streaming / Deployment / Visualization sets
- `pyproject.toml` + `requirements*.txt` cover every locked library
- `gamma_squeeze.stack.inventory.LOCKED_STACK` matches this document
- Orchestrator exposes `/v1/stack/inventory` and `/v1/stack/health`
- Docker (compose) + Kubernetes (`deploy/k8s`) + Next.js viz deps are documented and present
- Stack tests pass without requiring GPU or live infra
