"""Unified REST API surface mounted on the orchestrator gateway."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from services.common.schemas import ServiceEnvelope

REST_ENDPOINTS: tuple[str, ...] = (
    "/options",
    "/features",
    "/gex",
    "/gamma",
    "/regime",
    "/prediction",
    "/tft",
    "/xgboost",
    "/ppo",
    "/hedging",
    "/dashboard",
    "/alerts",
    "/backtest",
    "/train",
    "/retrain",
    "/health",
    "/docs",
)

router = APIRouter(tags=["rest"])


class SymbolBody(BaseModel):
    symbol: str
    as_of: str | None = None
    rebuild: bool = False


class TrainBody(BaseModel):
    symbols: list[str] | str = Field(..., description="Tickers")
    rebuild: bool = False
    n_trials: int = 15
    n_splits: int = 4
    validation_method: str = "walk_forward"
    force_retrain: bool = False
    use_mlflow: bool = True


class RetrainBody(BaseModel):
    symbols: list[str] | str
    backend: str = "baseline"
    rebuild_features: bool = True
    rebuild_labels: bool = True
    export_infer: bool = True
    mode: str = "baseline"  # baseline | rl | deep | training


class BacktestBody(BaseModel):
    symbols: list[str] | str
    n_splits: int = 4
    validation_method: str = "walk_forward"
    threshold: float = 0.5
    alpha: float = 0.05
    rebuild: bool = False


def _syms(symbols: list[str] | str) -> list[str]:
    if isinstance(symbols, str):
        return [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return [s.strip().upper() for s in symbols if s and str(s).strip()]


def _envelope(service: str, data: dict[str, Any], *, symbol: str | None = None) -> ServiceEnvelope:
    err = data.get("error") or (data.get("status") if data.get("status") not in (None, "ok") else None)
    # status like no_data is degraded but not always "error" key
    degraded = bool(data.get("error")) or (
        isinstance(data.get("status"), str) and data["status"] not in ("ok", "success", None)
        and data.get("success") is not True
    )
    # many handlers use success=False
    if data.get("success") is False:
        degraded = True
    upstream = []
    if data.get("error"):
        upstream.append(str(data["error"]))
    elif degraded and isinstance(data.get("status"), str):
        upstream.append(str(data["status"]))
    return ServiceEnvelope(
        service=service,
        symbol=(symbol or data.get("symbol") or (data.get("symbols") or [None])[0]),
        as_of=data.get("as_of"),
        degraded=degraded,
        upstream_errors=upstream,
        data=data,
    )


# ── handlers (in-process) ───────────────────────────────────────────────


def handle_options(symbol: str, *, as_of: str | None = None) -> dict[str, Any]:
    from services.data_collection.service import collect_snapshot
    from services.feature_engineering.service import compute_options_metrics_features

    snap = collect_snapshot(symbol, as_of=as_of)
    metrics = compute_options_metrics_features(symbol, as_of=as_of or snap.get("as_of"))
    return {
        "symbol": symbol.upper(),
        "as_of": metrics.get("as_of") or snap.get("as_of"),
        "collection": snap,
        "options_metrics": metrics.get("options_metrics") or metrics,
        "success": metrics.get("success", True) and "error" not in snap,
    }


def handle_features(symbol: str, *, rebuild: bool = False) -> dict[str, Any]:
    from services.feature_engineering.service import build_or_load_features

    return build_or_load_features(symbol, rebuild=rebuild)


def handle_gex(symbol: str, *, as_of: str | None = None) -> dict[str, Any]:
    from gamma_squeeze.config import alpaca_configured, resolve_matrix_root
    from gamma_squeeze.features.gex_features import gex_feature_row
    from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix

    sym = symbol.upper()
    matrix = None
    source = "none"
    date_used = as_of
    if alpaca_configured():
        try:
            from gamma_squeeze.ingest.alpaca_api import fetch_live_options_matrix

            live = fetch_live_options_matrix(sym, as_of=as_of)
            if live.get("contracts"):
                matrix = live
                source = "alpaca_api"
                date_used = live.get("as_of_date") or as_of
        except Exception:  # noqa: BLE001
            pass
    if matrix is None:
        root = resolve_matrix_root()
        dates = list_local_dates(root, sym)
        date_used = date_used or (dates[-1] if dates else None)
        if date_used:
            matrix = load_local_matrix(root, sym, date_used)
            if matrix:
                source = "ssd"
    if not matrix:
        return {"symbol": sym, "as_of": date_used, "error": "no_matrix", "source": source}
    try:
        row = gex_feature_row(matrix, as_of_date=str(date_used or ""))
        return {"symbol": sym, "as_of": date_used, "source": source, "gex": row, "success": True}
    except Exception as exc:  # noqa: BLE001
        return {"symbol": sym, "as_of": date_used, "error": str(exc), "source": source}


def handle_gamma(symbol: str, *, persist: bool = False) -> dict[str, Any]:
    from services.gamma_squeeze_engine.service import run_squeeze

    return run_squeeze(symbol, persist=persist, composite=True)


def handle_regime(symbol: str, *, train: bool = True) -> dict[str, Any]:
    from services.regime_hmm.service import run_regime

    return run_regime(symbol, train=train)


def handle_prediction(symbol: str, *, persist: bool = True) -> dict[str, Any]:
    from services.gamma_squeeze_engine.service import run_squeeze

    out = run_squeeze(symbol, persist=persist, composite=True)
    # Normalize prediction view
    return {
        "symbol": symbol.upper(),
        "as_of": out.get("as_of"),
        "prediction": out.get("forecast") or out.get("composite") or out,
        "engine": out,
        "success": "error" not in out,
    }


def handle_tft(symbol: str, *, train: bool = True) -> dict[str, Any]:
    from services.tft_forecasting.service import run_tft

    return run_tft(symbol, train=train)


def handle_xgboost(symbol: str, *, train: bool = True) -> dict[str, Any]:
    from services.xgboost_direction.service import run_direction

    return run_direction(symbol, train=train)


def handle_ppo(symbol: str, *, train: bool = False) -> dict[str, Any]:
    from services.ppo_agent.service import run_ppo

    return run_ppo(symbol, train=train)


def handle_hedging(symbol: str, *, optimize_for: str = "Risk") -> dict[str, Any]:
    from services.dealer_hedging.service import run_dealer_hedge
    from services.portfolio_hedging.service import run_portfolio_hedge

    dealer = run_dealer_hedge(symbol)
    portfolio = run_portfolio_hedge(symbol, optimize_for=optimize_for)
    return {
        "symbol": symbol.upper(),
        "as_of": portfolio.get("as_of") or dealer.get("as_of"),
        "dealer_hedging": dealer,
        "portfolio_hedging": portfolio,
        "success": "error" not in portfolio,
    }


def handle_dashboard(symbol: str) -> dict[str, Any]:
    from services.dashboard.service import build_dashboard

    return build_dashboard(symbol)


def handle_alerts(symbol: str) -> dict[str, Any]:
    from services.trade_alerts.service import evaluate_alerts

    return evaluate_alerts(symbol)


def handle_backtest(body: BacktestBody) -> dict[str, Any]:
    from gamma_squeeze.metrics.tracker import track_symbol_metrics
    from gamma_squeeze.training.pipeline import run_model_training

    symbols = _syms(body.symbols)
    train = run_model_training(
        symbols,
        rebuild=body.rebuild,
        n_trials=8,
        n_splits=body.n_splits,
        validation_method=body.validation_method,
        use_mlflow=False,
        auto_retrain=False,
        force_retrain=False,
    )
    metrics = track_symbol_metrics(
        symbols,
        rebuild=body.rebuild,
        threshold=body.threshold,
        alpha=body.alpha,
    )
    return {
        "symbols": symbols,
        "status": "ok" if train.status in ("ok", "insufficient_data") or metrics.status == "ok" else "error",
        "validation": train.validation,
        "hyperopt": {"status": train.hyperopt.get("status"), "best_value": train.hyperopt.get("best_value")},
        "metrics": metrics.to_dict(),
        "as_of": None,
    }


def handle_train(body: TrainBody) -> dict[str, Any]:
    from services.model_training.service import run_training

    return run_training(
        body.symbols,
        rebuild=body.rebuild,
        n_trials=body.n_trials,
        n_splits=body.n_splits,
        validation_method=body.validation_method,
        use_mlflow=body.use_mlflow,
        auto_retrain=True,
        force_retrain=body.force_retrain,
    )


def handle_retrain(body: RetrainBody) -> dict[str, Any]:
    symbols = _syms(body.symbols)
    mode = body.mode or body.backend
    if mode == "training":
        from services.model_training.service import run_training

        return run_training(
            symbols,
            rebuild=body.rebuild_features,
            force_retrain=True,
            auto_retrain=True,
        )
    from gamma_squeeze.retrain.pipeline import run_retrain

    result = run_retrain(
        symbols,
        rebuild_features=body.rebuild_features,
        rebuild_labels=body.rebuild_labels,
        backend=mode if mode in ("baseline", "rl", "deep") else "baseline",
        export_infer=body.export_infer,
    )
    return {
        "symbols": result.symbols,
        "metrics": result.metrics,
        "model_versions": result.model_versions,
        "forecast_paths": result.forecast_paths,
        "status": "ok" if "error" not in result.metrics else result.metrics.get("error"),
    }


# ── catalog ─────────────────────────────────────────────────────────────

_SYMBOL_ROOTS: tuple[str, ...] = (
    "/options",
    "/features",
    "/gex",
    "/gamma",
    "/regime",
    "/prediction",
    "/tft",
    "/xgboost",
    "/ppo",
    "/hedging",
    "/dashboard",
    "/alerts",
)


def _route_help(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "methods": ["GET", "POST"],
        "usage": {
            "GET": f"{path}/{{symbol}}",
            "POST": {"body": {"symbol": "AAPL"}},
        },
        "docs": "/docs",
        "catalog": "/api",
    }


@router.get("/api")
def api_catalog() -> dict[str, Any]:
    return {
        "service": "orchestrator",
        "phase": 18,
        "implement": list(REST_ENDPOINTS),
        "note": "/health and /docs are provided by the FastAPI app factory / Swagger UI",
        "symbol_routes": [f"{p}/{{symbol}}" for p in _SYMBOL_ROOTS],
        "job_routes": ["/backtest", "/train", "/retrain"],
        "ops_routes": ["/health", "/docs"],
    }


# ── REST routes ─────────────────────────────────────────────────────────


@router.get("/options")
def options_root() -> dict[str, Any]:
    return _route_help("/options")


@router.get("/options/{symbol}", response_model=ServiceEnvelope)
def options_get(symbol: str, as_of: str | None = None) -> ServiceEnvelope:
    return _envelope("options", handle_options(symbol, as_of=as_of), symbol=symbol.upper())


@router.post("/options", response_model=ServiceEnvelope)
def options_post(req: SymbolBody) -> ServiceEnvelope:
    return options_get(req.symbol, as_of=req.as_of)


@router.get("/features")
def features_root() -> dict[str, Any]:
    return _route_help("/features")


@router.get("/features/{symbol}", response_model=ServiceEnvelope)
def features_get(symbol: str, rebuild: bool = False) -> ServiceEnvelope:
    return _envelope("features", handle_features(symbol, rebuild=rebuild), symbol=symbol.upper())


@router.post("/features", response_model=ServiceEnvelope)
def features_post(req: SymbolBody) -> ServiceEnvelope:
    return features_get(req.symbol, rebuild=req.rebuild)


@router.get("/gex")
def gex_root() -> dict[str, Any]:
    return _route_help("/gex")


@router.get("/gex/{symbol}", response_model=ServiceEnvelope)
def gex_get(symbol: str, as_of: str | None = None) -> ServiceEnvelope:
    return _envelope("gex", handle_gex(symbol, as_of=as_of), symbol=symbol.upper())


@router.post("/gex", response_model=ServiceEnvelope)
def gex_post(req: SymbolBody) -> ServiceEnvelope:
    return gex_get(req.symbol, as_of=req.as_of)


@router.get("/gamma")
def gamma_root() -> dict[str, Any]:
    return _route_help("/gamma")


@router.get("/gamma/{symbol}", response_model=ServiceEnvelope)
def gamma_get(symbol: str, persist: bool = False) -> ServiceEnvelope:
    return _envelope("gamma", handle_gamma(symbol, persist=persist), symbol=symbol.upper())


@router.post("/gamma", response_model=ServiceEnvelope)
def gamma_post(req: SymbolBody) -> ServiceEnvelope:
    return gamma_get(req.symbol)


@router.get("/regime")
def regime_root() -> dict[str, Any]:
    return _route_help("/regime")


@router.get("/regime/{symbol}", response_model=ServiceEnvelope)
def regime_get(symbol: str, train: bool = True) -> ServiceEnvelope:
    return _envelope("regime", handle_regime(symbol, train=train), symbol=symbol.upper())


@router.post("/regime", response_model=ServiceEnvelope)
def regime_post(req: SymbolBody) -> ServiceEnvelope:
    return regime_get(req.symbol)


@router.get("/prediction")
def prediction_root() -> dict[str, Any]:
    return _route_help("/prediction")


@router.get("/prediction/{symbol}", response_model=ServiceEnvelope)
def prediction_get(symbol: str, persist: bool = True) -> ServiceEnvelope:
    return _envelope("prediction", handle_prediction(symbol, persist=persist), symbol=symbol.upper())


@router.post("/prediction", response_model=ServiceEnvelope)
def prediction_post(req: SymbolBody) -> ServiceEnvelope:
    return prediction_get(req.symbol)


@router.get("/tft")
def tft_root() -> dict[str, Any]:
    return _route_help("/tft")


@router.get("/tft/{symbol}", response_model=ServiceEnvelope)
def tft_get(symbol: str, train: bool = Query(True)) -> ServiceEnvelope:
    return _envelope("tft", handle_tft(symbol, train=train), symbol=symbol.upper())


@router.post("/tft", response_model=ServiceEnvelope)
def tft_post(req: SymbolBody) -> ServiceEnvelope:
    return tft_get(req.symbol)


@router.get("/xgboost")
def xgboost_root() -> dict[str, Any]:
    return _route_help("/xgboost")


@router.get("/xgboost/{symbol}", response_model=ServiceEnvelope)
def xgboost_get(symbol: str, train: bool = Query(True)) -> ServiceEnvelope:
    return _envelope("xgboost", handle_xgboost(symbol, train=train), symbol=symbol.upper())


@router.post("/xgboost", response_model=ServiceEnvelope)
def xgboost_post(req: SymbolBody) -> ServiceEnvelope:
    return xgboost_get(req.symbol)


@router.get("/ppo")
def ppo_root() -> dict[str, Any]:
    return _route_help("/ppo")


@router.get("/ppo/{symbol}", response_model=ServiceEnvelope)
def ppo_get(symbol: str, train: bool = False) -> ServiceEnvelope:
    return _envelope("ppo", handle_ppo(symbol, train=train), symbol=symbol.upper())


@router.post("/ppo", response_model=ServiceEnvelope)
def ppo_post(req: SymbolBody) -> ServiceEnvelope:
    return ppo_get(req.symbol)


@router.get("/hedging")
def hedging_root() -> dict[str, Any]:
    return _route_help("/hedging")


@router.get("/hedging/{symbol}", response_model=ServiceEnvelope)
def hedging_get(symbol: str, optimize_for: str = "Risk") -> ServiceEnvelope:
    return _envelope("hedging", handle_hedging(symbol, optimize_for=optimize_for), symbol=symbol.upper())


@router.post("/hedging", response_model=ServiceEnvelope)
def hedging_post(req: SymbolBody) -> ServiceEnvelope:
    return hedging_get(req.symbol)


@router.get("/dashboard")
def dashboard_root() -> dict[str, Any]:
    return _route_help("/dashboard")


@router.get("/dashboard/{symbol}", response_model=ServiceEnvelope)
def dashboard_get(symbol: str) -> ServiceEnvelope:
    return _envelope("dashboard", handle_dashboard(symbol), symbol=symbol.upper())


@router.post("/dashboard", response_model=ServiceEnvelope)
def dashboard_post(req: SymbolBody) -> ServiceEnvelope:
    return dashboard_get(req.symbol)


@router.get("/alerts")
def alerts_root() -> dict[str, Any]:
    return _route_help("/alerts")


@router.get("/alerts/{symbol}", response_model=ServiceEnvelope)
def alerts_get(symbol: str) -> ServiceEnvelope:
    return _envelope("alerts", handle_alerts(symbol), symbol=symbol.upper())


@router.post("/alerts", response_model=ServiceEnvelope)
def alerts_post(req: SymbolBody) -> ServiceEnvelope:
    return alerts_get(req.symbol)


@router.post("/backtest", response_model=ServiceEnvelope)
def backtest_post(req: BacktestBody) -> ServiceEnvelope:
    data = handle_backtest(req)
    return _envelope("backtest", data, symbol=_syms(req.symbols)[0] if _syms(req.symbols) else None)


@router.get("/backtest", response_model=ServiceEnvelope)
def backtest_get(
    symbols: str = Query(..., description="Comma-separated tickers"),
    n_splits: int = 4,
    validation_method: str = "walk_forward",
) -> ServiceEnvelope:
    return backtest_post(
        BacktestBody(symbols=symbols, n_splits=n_splits, validation_method=validation_method)
    )


@router.post("/train", response_model=ServiceEnvelope)
def train_post(req: TrainBody) -> ServiceEnvelope:
    data = handle_train(req)
    syms = data.get("symbols") or _syms(req.symbols)
    return _envelope("train", data, symbol=syms[0] if syms else None)


@router.get("/train", response_model=ServiceEnvelope)
def train_get(symbols: str = Query(...), n_trials: int = 15, force_retrain: bool = False) -> ServiceEnvelope:
    return train_post(TrainBody(symbols=symbols, n_trials=n_trials, force_retrain=force_retrain))


@router.post("/retrain", response_model=ServiceEnvelope)
def retrain_post(req: RetrainBody) -> ServiceEnvelope:
    data = handle_retrain(req)
    syms = data.get("symbols") or _syms(req.symbols)
    return _envelope("retrain", data, symbol=syms[0] if syms else None)


@router.get("/retrain", response_model=ServiceEnvelope)
def retrain_get(
    symbols: str = Query(...),
    backend: str = "baseline",
    mode: str = "baseline",
) -> ServiceEnvelope:
    return retrain_post(RetrainBody(symbols=symbols, backend=backend, mode=mode))
