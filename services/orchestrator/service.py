"""Pipeline orchestration — in-process or HTTP fan-out with per-stage status."""

from __future__ import annotations

from typing import Any, Callable, Literal

from services.common.http_client import get_json
from services.common.pipeline import (
    COMPANION_STEPS,
    CORE_PIPELINE_STEPS,
    HTTP_STAGE_PATHS,
    LOCKED_PORTS,
    PIPELINE_LABELS,
    PIPELINE_STEPS,
)
from services.common.registry import service_url
from services.dashboard.service import build_dashboard
from services.data_collection.service import collect_snapshot
from services.dealer_hedging.service import run_dealer_hedge
from services.explainability.service import run_explain
from services.feature_engineering.service import build_or_load_features
from services.gamma_squeeze_engine.service import run_squeeze
from services.model_training.service import run_training
from services.pattern_recognition.service import run_patterns
from services.performance_metrics.service import track_symbols
from services.portfolio_hedging.service import run_portfolio_hedge
from services.ppo_agent.service import run_ppo
from services.regime_hmm.service import run_regime
from services.tft_forecasting.service import run_tft
from services.trade_alerts.service import evaluate_alerts
from services.xgboost_direction.service import run_direction

PipelineMode = Literal["in_process", "http"]


def pipeline_contract() -> dict[str, Any]:
    """Document the locked DAG for OpenAPI / registry consumers."""
    return {
        "core_pipeline": [
            {
                "step": name,
                "label": PIPELINE_LABELS.get(name, name),
                "port": LOCKED_PORTS[name],
                "http_path": HTTP_STAGE_PATHS.get(name),
            }
            for name in CORE_PIPELINE_STEPS
        ],
        "companions": list(COMPANION_STEPS),
        "modes": ["in_process", "http"],
        "independence": (
            "Each stage is an independent FastAPI service; stage failure does not "
            "abort the DAG — later stages still run and report status."
        ),
    }


def _run_step(
    name: str,
    fn: Callable[[], Any],
    *,
    results: dict[str, Any],
    stages: list[dict[str, Any]],
) -> Any:
    """Execute one stage; never raise — record ok/error and continue."""
    label = PIPELINE_LABELS.get(name, name)
    try:
        out = fn()
        results[name] = out
        stages.append(
            {
                "step": name,
                "label": label,
                "status": "ok",
                "ok": True,
                "error": None,
            }
        )
        return out
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        results[name] = {"error": err}
        stages.append(
            {
                "step": name,
                "label": label,
                "status": "error",
                "ok": False,
                "error": err,
            }
        )
        return None


def _run_in_process(
    symbol: str,
    *,
    rebuild_features: bool,
    include_companions: bool,
) -> dict[str, Any]:
    sym = symbol.upper()
    stages: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    def step(name: str, fn: Callable[[], Any]) -> Any:
        return _run_step(name, fn, results=results, stages=stages)

    step("data_collection", lambda: collect_snapshot(sym))
    step(
        "feature_engineering",
        lambda: build_or_load_features(sym, rebuild=rebuild_features),
    )
    step("pattern_recognition", lambda: run_patterns(sym))
    step("regime_hmm", lambda: run_regime(sym, train=True))
    step("xgboost_direction", lambda: run_direction(sym, train=True))
    step("tft_forecasting", lambda: run_tft(sym, train=True))
    step("dealer_hedging", lambda: run_dealer_hedge(sym))
    squeeze = step("gamma_squeeze_engine", lambda: run_squeeze(sym, persist=True))
    step("ppo_agent", lambda: run_ppo(sym))
    step("portfolio_hedging", lambda: run_portfolio_hedge(sym))
    step("dashboard", lambda: build_dashboard(sym))
    step("trade_alerts", lambda: evaluate_alerts(sym))

    if include_companions:
        step("explainability", lambda: run_explain(sym, include_forecast=True))
        step(
            "model_training",
            lambda: run_training(
                [sym],
                rebuild=False,
                n_trials=8,
                n_splits=3,
                use_mlflow=True,
                auto_retrain=True,
                force_retrain=False,
            ),
        )
        step("performance_metrics", lambda: track_symbols([sym], rebuild=False))

    as_of = squeeze.get("as_of") if isinstance(squeeze, dict) else None
    steps = list(CORE_PIPELINE_STEPS) + (
        list(COMPANION_STEPS) if include_companions else []
    )
    return _finalize(sym, as_of, steps, stages, results, mode="in_process")


def _http_call(service: str, symbol: str, *, timeout: float = 60.0) -> dict[str, Any]:
    path_tmpl = HTTP_STAGE_PATHS[service]
    path = path_tmpl.format(symbol=symbol.upper())
    url = service_url(service) + path
    return get_json(url, timeout=timeout)


def _run_http(
    symbol: str,
    *,
    include_companions: bool,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Ordered HTTP fan-out to peer services; stage failures do not abort the DAG."""
    sym = symbol.upper()
    stages: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    steps = list(CORE_PIPELINE_STEPS) + (
        list(COMPANION_STEPS) if include_companions else []
    )

    squeeze: Any = None
    for name in steps:
        out = _run_step(
            name,
            lambda n=name: _http_call(n, sym, timeout=timeout),
            results=results,
            stages=stages,
        )
        if name == "gamma_squeeze_engine":
            squeeze = out

    as_of = None
    if isinstance(squeeze, dict):
        data = squeeze.get("data") if "data" in squeeze else squeeze
        if isinstance(data, dict):
            as_of = data.get("as_of") or squeeze.get("as_of")
        else:
            as_of = squeeze.get("as_of")

    return _finalize(sym, as_of, steps, stages, results, mode="http")


def _finalize(
    symbol: str,
    as_of: Any,
    steps: list[str],
    stages: list[dict[str, Any]],
    results: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    # Backward-compatible "trace" alias used by existing API envelope
    trace = [
        {"step": s["step"], "ok": s["ok"], "error": s["error"]} for s in stages
    ]
    core_ok = all(
        s["ok"] for s in stages if s["step"] in CORE_PIPELINE_STEPS
    )
    return {
        "symbol": symbol,
        "as_of": as_of,
        "mode": mode,
        "steps": steps,
        "core_pipeline": list(CORE_PIPELINE_STEPS),
        "stages": stages,
        "trace": trace,
        "results": results,
        "ok": all(s["ok"] for s in stages),
        "core_ok": core_ok,
        "contract": pipeline_contract(),
    }


def run_pipeline(
    symbol: str,
    *,
    rebuild_features: bool = False,
    mode: PipelineMode = "in_process",
    include_companions: bool = True,
    http_timeout: float = 60.0,
) -> dict[str, Any]:
    """
    Execute the ordered microservice DAG.

    - ``in_process``: call domain service modules directly (default; works without peers).
    - ``http``: GET each peer's primary endpoint via registry URLs.

    Stage failures are recorded; the DAG continues so callers always get per-stage status.
    """
    if mode == "http":
        return _run_http(
            symbol,
            include_companions=include_companions,
            timeout=http_timeout,
        )
    return _run_in_process(
        symbol,
        rebuild_features=rebuild_features,
        include_companions=include_companions,
    )


# Re-export for tests / callers that imported PIPELINE_STEPS from this module
__all__ = [
    "PIPELINE_STEPS",
    "CORE_PIPELINE_STEPS",
    "COMPANION_STEPS",
    "pipeline_contract",
    "run_pipeline",
]
