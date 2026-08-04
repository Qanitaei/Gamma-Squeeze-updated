# Gamma Squeeze Platform

Modular institutional platform for forecasting **gamma squeezes 1–10 trading days** ahead: calibrated probability, magnitude, duration, return distribution, dealer hedging demand, risk-adjusted trades, portfolio hedges, confidence, and explainability.

Consumes **Alpaca** for options/OHLCV (live API + backup KV worker) and shared workers (FRED / calendar / skew). GEX/DEX and Alpaca clients are vendored in-platform — **no sibling data-plane dependency**; dashboard is JSON-only (no HTML desk).

**Microservices architecture:** see [ARCHITECTURE.md](ARCHITECTURE.md), [docs/PROGRAMMING_STACK.md](docs/PROGRAMMING_STACK.md), [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md), [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md) (DI, typing, async, YAML+env config, structured logging/metrics, swappable models, reproducibility).

**Stack (locked):** Python **3.12+** · PyTorch / Lightning · XGBoost / LightGBM / CatBoost · SB3 / Ray RLlib · Darts / PyTorch Forecasting · sklearn / SHAP / Optuna / MLflow · FastAPI · Redis · PostgreSQL · DuckDB · Arrow / Polars · Kafka · WebSockets · Docker / K8s · React / Next.js · Plotly · TradingView Lightweight Charts · ECharts.

## Phased delivery

| Phase | Topic |
|-------|--------|
| **1 — System Architecture** | Independent FastAPI microservices, locked DAG, Docker Compose ([ARCHITECTURE.md](ARCHITECTURE.md)) |
| **2 — Programming Stack** | Python 3.12+, ML/RL/forecast, FastAPI/Redis/Postgres/DuckDB/Arrow/Polars, Kafka/WS, Docker/K8s, React/Next/Plotly/TV/ECharts ([docs/PROGRAMMING_STACK.md](docs/PROGRAMMING_STACK.md)) |
| **3 — Data Sources** | Options (Alpaca API + `alpaca-options-matrix-backup` KV; IBKR/OPRA/Polygon/CBOE), Market, Macro, Structure, Corporate, Calendar ([docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)) |
| **4 — Feature Engineering** | Dealer Positioning + Options Metrics + Technical Indicators + Macro Features ([docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md)) |
| **5 — Hidden Markov Model** | 16 regimes (Bull…Panic); current state, transition matrix, next-state probs, confidence ([docs/REGIME_HMM.md](docs/REGIME_HMM.md)) |
| **6 — XGBoost Direction** | 1/3/5/10/20d returns — classification, regression, probability; CV + Bayesian Optuna + SHAP ([docs/XGBOOST_DIRECTION.md](docs/XGBOOST_DIRECTION.md)) |
| **7 — Temporal Fusion Transformer** | Future Price/IV/Net GEX/Hedge/Flip/Walls/EM — 1/2/3/5/10d median + q10–q90 + attention ([docs/TFT_FORECASTING.md](docs/TFT_FORECASTING.md)) |
| **8 — Technical Pattern Recognition** | CNN/LSTM/Transformer hybrid — flags/triangles/HS/candles; pattern + confidence + target + probability ([docs/PATTERN_RECOGNITION.md](docs/PATTERN_RECOGNITION.md)) |
| **9 — Dealer Hedging Engine** | Share buys/sales, gamma ramp, migration, inventory/liquidity/exhaustion/flip → flow map, demand curve, hedge volume ([docs/DEALER_HEDGING.md](docs/DEALER_HEDGING.md)) |
| **10 — Gamma Squeeze Engine** | Composite scores → Probability, Magnitude, Duration, Start, Confidence, Risk Rating ([docs/GAMMA_SQUEEZE_ENGINE.md](docs/GAMMA_SQUEEZE_ENGINE.md)) |
| **11 — PPO RL Agent** | Stable-Baselines3 PPO — dealer/regime/forecast/… state; 13 structures/hedge actions; Sharpe/Sortino/DD/costs reward ([docs/PPO_AGENT.md](docs/PPO_AGENT.md)) |
| **12 — Portfolio Hedging** | Long/Puts/spreads/condor/collar + Δ/Γ/ν/Θ hedges; optimize Return · Risk · Capital Efficiency · Margin ([docs/PORTFOLIO_HEDGING.md](docs/PORTFOLIO_HEDGING.md)) |
| **13 — Explainability** | SHAP · Attention · Feature Importance · Counterfactuals · PDP · Decision Trace · Confidence — every prediction has WHY ([docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md)) |
| 14 | Visualization Dashboard | Deferred |
| **15 — Alert Engine** | Squeeze P / dealer flip / gamma ramp / walls / IV / hedge / patterns / earnings → WS·Email·SMS·Discord·Slack·Webhook ([docs/ALERT_ENGINE.md](docs/ALERT_ENGINE.md)) |
| **16 — Model Training** | Walk-forward / TSS · Bayesian HPO · Early Stopping · MLflow · Versioning · Auto-retrain · Feature/Prediction/Concept Drift ([docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md)) |
| **17 — Performance Metrics** | Accuracy·Precision·Recall·F1·AUC·Brier·LogLoss · MAE/RMSE · Sharpe/Sortino/Calmar/DD/Return · Win/PF/Trade/Hold · Tail/ES ([docs/PERFORMANCE_METRICS.md](docs/PERFORMANCE_METRICS.md)) |
| **18 — API Endpoints** | REST gateway: `/options` `/features` `/gex` `/gamma` `/regime` `/prediction` `/tft` `/xgboost` `/ppo` `/hedging` `/dashboard` `/alerts` `/backtest` `/train` `/retrain` `/health` `/docs` ([docs/API_ENDPOINTS.md](docs/API_ENDPOINTS.md)) |
| **19 — Coding Standards** | Clean architecture + DI · strong typing · unit/integration tests · async APIs · swappable models · YAML+env config · structured logging/metrics · docs · reproducible pipelines ([docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md)) |

**Phase 1 core pipeline:** Data Collection → Feature Engineering → Pattern Recognition → HMM → XGBoost → TFT → Dealer Hedging → Gamma Squeeze Engine → PPO → Portfolio Hedging → Dashboard → Trade Alert API. Each module exposes an independent API and runs alone.

## Layout

```
gamma-squeeze-platform/
  ARCHITECTURE.md
  docker-compose.yml     # one container per microservice (8000–8015)
  Dockerfile             # shared image; SERVICE + PORT select entrypoint
  docs/                  # stack, data sources, coding standards, metrics
  services/              # independent FastAPI microservices (ports 8000–8015)
  src/gamma_squeeze/     # domain libraries (no HTTP)
  scripts/               # CLI + run_all_services.py
  cloudflare-worker-squeeze/
  tests/
```

## Microservices (independent APIs)

| Port | Service | Key endpoints |
|------|---------|---------------|
| 8000 | Orchestrator | Unified REST `/options`…`/retrain` + `/v1/pipeline/run`, `/docs`, `/health` |
| 8001 | Data Collection | `/v1/collect/{symbol}` |
| 8002 | Feature Engineering | `/v1/features/{symbol}` |
| 8003 | Pattern Recognition | `/v1/patterns/{symbol}`, `/v1/catalog` |
| 8004 | HMM Regime | `/v1/regime/{symbol}`, `/v1/states` |
| 8005 | XGBoost Direction | `/v1/direction/{symbol}`, `/v1/horizons` |
| 8006 | TFT Forecasting | `/v1/forecast/{symbol}`, `/v1/meta` |
| 8007 | Dealer Hedging | `/v1/hedge-demand/{symbol}`, `/v1/simulate`, `/v1/meta` |
| 8008 | Gamma Squeeze Engine | `/v1/squeeze/{symbol}`, `/v1/meta` |
| 8009 | PPO Agent | `/v1/action/{symbol}`, `/v1/train`, `/v1/meta` |
| 8010 | Portfolio Hedging | `/v1/hedges/{symbol}`, `/v1/optimize`, `/v1/meta` |
| 8011 | Dashboard | `/v1/dashboard/{symbol}`, `/v1/meta` (Alpaca health, JSON panels) |
| 8012 | Alert Engine | `/v1/alerts/{symbol}`, `/v1/notify`, `/v1/ws/alerts`, `/v1/meta` |
| 8013 | Explainability | `/v1/explain/{symbol}` (meta via Alpaca API health) |
| 8014 | Model Training | `/v1/train`, `/v1/versions`, `/v1/rollback`, `/v1/meta` |
| 8015 | Performance Metrics | `/v1/track/{symbol}`, `/v1/compute`, `/v1/meta` |

```bash
# one service
PYTHONPATH=.:src uvicorn services.data_collection.app:app --port 8001

# all services (local processes)
python scripts/run_all_services.py

# all services (Docker — healthchecks on /health)
docker compose up -d --build

# full pipeline via gateway (in-process composition)
PYTHONPATH=.:src uvicorn services.orchestrator.app:app --port 8000
curl 'http://127.0.0.1:8000/v1/pipeline/run?symbol=AAPL'
curl 'http://127.0.0.1:8000/v1/pipeline/run?symbol=AAPL&mode=http'

# unified REST (see docs/API_ENDPOINTS.md)
curl 'http://127.0.0.1:8000/api'
curl 'http://127.0.0.1:8000/features/AAPL'
curl 'http://127.0.0.1:8000/prediction/AAPL'
curl 'http://127.0.0.1:8000/docs'
```

Each service exposes `/health`, `/ready`, `/docs`, `/openapi.json` and can run alone (see `services/<name>/README.md`). Gateway REST: `/options`, `/features`, `/gex`, `/gamma`, `/regime`, `/prediction`, `/tft`, `/xgboost`, `/ppo`, `/hedging`, `/dashboard`, `/alerts`, `/backtest`, `/train`, `/retrain`.

## Setup

```bash
cd gamma-squeeze-platform
# Prefer Python 3.12+
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional Phase-2 layers:
# pip install -r requirements-ml.txt -r requirements-rl.txt -r requirements-forecast.txt \
#   -r requirements-explain.txt -r requirements-train.txt -r requirements-streaming.txt -r requirements-viz.txt
# or: pip install -e ".[ml,rl,forecast,explain,train,streaming,viz]"
cp .env.example .env

# infra (Redis, Postgres, Redpanda/Kafka, MLflow)
docker compose -f docker-compose.infra.yml up -d

# microservices (Docker)
docker compose up -d --build

# Kubernetes (optional)
# kubectl apply -f deploy/k8s/

# Next.js visualization stack (React / Plotly / Lightweight Charts / ECharts)
cd apps/web-dashboard && npm install && npm run dev

# stack inventory API (orchestrator)
# curl -s http://127.0.0.1:8000/v1/stack/inventory | jq .
# curl -s http://127.0.0.1:8000/v1/stack/health | jq .
```

## Annual options → Portable SSD

Pull history from KV namespace **`alpaca-options-matrix-backup`** into:

`/Volumes/PortableSSD/Gamma Squeeze Matrix/`

```bash
python scripts/export_gamma_squeeze_matrix_from_kv.py --resume
# smoke:
python scripts/export_gamma_squeeze_matrix_from_kv.py --symbols AAPL,MSFT --lookback-days 60 --resume
```

Layout:

```
Gamma Squeeze Matrix/
  index.json
  state.json
  tickers/{SYMBOL}/{YYYY-MM-DD}.json
  manifests/{SYMBOL}.json
  features/  models/  forecasts/   # created by later steps
```

Cloud / unmounted SSD falls back to `data/exports/Gamma Squeeze Matrix/`.

## Research runbook (features → train → infer)

```bash
# 1) Features (prefers local SSD matrices)
python scripts/run_feature_build.py --symbols AAPL,MSFT --no-skew --no-macro

# 2) Labels
python scripts/run_label_build.py --symbols AAPL,MSFT

# 3) Train baselines
python scripts/run_train.py --symbols AAPL,MSFT

# 4) Infer schema-complete forecasts
python scripts/run_infer.py --symbols AAPL,MSFT

# 5) Continuous retrain (+ optional rl|deep stubs)
python scripts/run_retrain_loop.py --symbols AAPL,MSFT --backend baseline
python scripts/run_retrain_loop.py --symbols AAPL,MSFT --backend rl --skip-features --skip-labels --no-infer

# 6) Model training (walk-forward / Bayesian / MLflow / drift)
python scripts/run_model_training.py --symbols AAPL,MSFT --n-trials 15 --validation walk_forward
python scripts/run_retrain_loop.py --symbols AAPL --mode training --n-trials 12 --no-infer
```

See [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md).


## Forecast schema

Each payload includes horizons `1..10` with:

- `squeeze_probability`, `expected_magnitude_pct`, `magnitude_quantiles`
- `expected_duration_days`, `return_distribution`, `confidence_score`
- plus top-level `dealer_hedging_demand`, `trade_recommendations`,
  `portfolio_hedge_recommendations`, `explanations`, `model_versions`

## Cloudflare Worker (gamma-squeeze-data)

KV namespace **`gamma-squeeze-data`** bound as `KV` on worker  
[https://gamma-squeeze-data-icy-shadow-db40.2s6m8rz8fc.workers.dev/](https://gamma-squeeze-data-icy-shadow-db40.2s6m8rz8fc.workers.dev/)  
OpenAPI **3.0.3** at `/` and `/openapi.json`.

```bash
PYTHONPATH=.:src python scripts/deploy_gamma_squeeze_data_worker.py
PYTHONPATH=.:src python scripts/upload_gamma_squeeze_data_kv.py
```

Endpoints: `/health`, `/openapi.json`, `/v1/scans/phase14/latest`, `/v1/{symbol}/pipeline/latest`, `/v1/{symbol}/forecast/latest`, `/v1/{symbol}/forecast/{date}`.

Keys: `index`, `scans/phase14_pipeline/latest`, `{TICKER}/pipeline/latest`, `{TICKER}/latest`, `{TICKER}/forecast/{date}`.

## Tests

```bash
pytest -q
```

## Notes

- RL (`rl/agent.py`) and deep TFT (`deep/temporal_model.py`) are **registered stubs** so the retrain loop can swap backends later.
- Live broker execution is out of scope for Phase 1–2.
