# Coding Standards (Phase 19)

Institutional quantitative research & execution environment for forecasting gamma squeezes, simulating dealer hedging, recommending option structures, dynamically hedging portfolios, and delivering explainable AI forecasts through an interactive dashboard.

Module: `src/gamma_squeeze/core/` · Catalog: `GET /v1/standards` · Config: `config/platform.yaml`

## Principles

| Standard | Implementation |
|----------|----------------|
| Clean architecture + DI | Domain in `src/gamma_squeeze/`; HTTP in `services/`; DI via `core.container.Container` |
| Strong typing | Python 3.12+ annotations, `Protocol` ports, `py.typed`, typed settings dataclasses |
| Unit + integration tests | `pytest` markers `@pytest.mark.unit` / `@pytest.mark.integration` |
| Async APIs where appropriate | FastAPI `async def` handlers; `core.async_http` for concurrent fan-out |
| Modular / swappable models | `core.ports` + `core.model_registry` (register/replace predictors) |
| Config via env + YAML | `config/platform.yaml` + `.env` (`PLATFORM_CONFIG_PATH`, `core.settings`) |
| Structured logging + metrics | JSON logs (`core.logging`); counters/timers (`core.metrics_collector`); `GET /metrics` |
| Documentation | `docs/*`, OpenAPI `/docs`, inline module docstrings |
| Deterministic / reproducible | `seed_everything`, `stable_hash`, sorted feature columns |

Constant: `CODING_STANDARDS` (9 standards) via `gamma_squeeze.core.standards`.

## Structure

- Python **3.12+** (`requires-python = ">=3.12"`)
- One microservice = `services/<name>/app.py` + `service.py`
- Domain logic in `src/gamma_squeeze/` (no HTTP)
- Stack adapters in `src/gamma_squeeze/stack/`
- Shared HTTP/schema utilities in `services/common/` only
- Platform foundations in `src/gamma_squeeze/core/`
- Visualization UI in `apps/web-dashboard` (Next.js)

```
services/          → adapters (FastAPI)
src/gamma_squeeze/ → domain + use-cases
core/              → DI, config, logging, metrics, ports, standards
config/            → YAML defaults
```

## Dependency injection

```python
from gamma_squeeze.core import get_container, load_platform_settings

c = get_container()
settings = c.resolve("settings")          # PlatformSettings
metrics = c.resolve("metrics")            # MetricsCollector
backend = c.get("model.squeeze")          # e.g. "composite"
```

Prefer constructor injection in new modules; use the container for process singletons.

## Modular models

```python
from gamma_squeeze.core.model_registry import (
    bootstrap_default_predictors,
    get_predictor,
    register_predictor,
    CallablePredictor,
)

bootstrap_default_predictors()
pred = get_predictor("squeeze")
pred.predict("AAPL")

# Replace without touching callers:
register_predictor("squeeze", lambda: CallablePredictor("my_model", my_fn))
```

Ports: `PredictorPort`, `TrainablePort`, `ExplainerPort`, `HedgerPort`, `FeatureBuilderPort`.

## Configuration

1. Load `config/platform.yaml` (or `PLATFORM_CONFIG_PATH`)
2. Overlay environment variables (`.env` via `python-dotenv`)
3. Access via `load_platform_settings()`

Key env vars: `GLOBAL_SEED`, `LOG_LEVEL`, `LOG_JSON`, `METRICS_ENABLED`, `FORECAST_MARKET_TZ`, stack URLs (`DATABASE_URL`, `REDIS_URL`, `MLFLOW_TRACKING_URI`, …). Never commit secrets.

## Logging & metrics

- JSON structured logs to stdout (`setup_logging`)
- Request middleware records latency + `http.requests`
- Snapshot: `GET /metrics` on every service

```python
from gamma_squeeze.core import get_logger, get_metrics
from gamma_squeeze.core.logging import log_event

log_event(get_logger("train"), "fold_done", service="model_training", fold=2, auc=0.71)
with get_metrics().timer("train.fit"):
    ...
```

## Async

- Prefer `async def` for I/O-bound gateway routes
- Use `aget_json` / `gather_health` for concurrent peer checks
- Keep CPU-bound ML in sync functions (threadpool / in-process) unless profiling says otherwise

## API conventions

- `GET /health` — liveness
- `GET /ready` — readiness
- `GET /metrics` — process metrics snapshot
- `GET /v1/standards` — locked coding-standards catalog
- `POST /v1/...` compute; `GET /v1/...` idempotent reads
- Gateway REST: `/options` … `/retrain` (see [API_ENDPOINTS.md](API_ENDPOINTS.md))
- Envelope: `service`, `version`, `symbol`, `as_of`, `degraded`, `upstream_errors[]`
- Independence: a service must answer `/health` even if upstreams are down

## Reproducibility

```python
from gamma_squeeze.core import seed_everything, stable_hash

seed = seed_everything()           # from YAML / GLOBAL_SEED
fingerprint = stable_hash(X)      # feature matrix fingerprint
```

Deterministic preprocessing: stable column ordering (`sorted_feature_matrix`), fixed seeds in Optuna/sklearn, MLflow params for experiment lineage.

## Testing

```bash
pytest -m unit
pytest -m integration
pytest -q   # full suite
```

- Unit: pure metrics, validation, DI, config parsing
- Integration: FastAPI `TestClient`, multi-module pipelines
- Smoke: every service `/health`
- Isolation: `tests/conftest.py` resets the DI container per test

## Style

- Type hints on all public functions
- Small pure functions; avoid god-modules
- Extensive module/docstring comments for institutional handoff
- Conventional commits if committing; never commit `.env`
- Ruff: `py312`, line length 110

## Product north star

The platform must support end-to-end:

1. Forecast gamma squeezes (1–10 trading days)
2. Simulate dealer hedging demand
3. Recommend optimal option structures
4. Dynamically hedge portfolios
5. Deliver fully explainable AI forecasts via the interactive dashboard

## Phase 19 complete criteria

- All 9 standards locked in `CODING_STANDARDS` / `standards_meta()`
- DI container, ports, YAML+env settings, JSON logging, metrics, reproducibility helpers live
- Unit + integration markers configured; standards tests pass
- `GET /v1/standards` exposes the catalog
- Documentation in `docs/CODING_STANDARDS.md`
