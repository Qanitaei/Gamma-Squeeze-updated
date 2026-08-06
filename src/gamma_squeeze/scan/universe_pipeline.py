"""Top market-cap universe × 14-stage product pipeline → PortableSSD export."""

from __future__ import annotations

import csv
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gamma_squeeze.config import resolve_matrix_root
from gamma_squeeze.core.logging import get_logger, log_event
from gamma_squeeze.core.reproducibility import seed_everything
from gamma_squeeze.scan.confirmed_squeeze import load_top_market_cap

logger = get_logger("scan.universe_pipeline")

# Product stages through Phase 14 dashboard + explain + metrics (skip heavy retrain)
UNIVERSE_PIPELINE_STEPS: tuple[str, ...] = (
    "data_collection",
    "feature_engineering",
    "pattern_recognition",
    "regime_hmm",
    "xgboost_direction",
    "tft_forecasting",
    "dealer_hedging",
    "gamma_squeeze_engine",
    "ppo_agent",
    "portfolio_hedging",
    "dashboard",
    "trade_alerts",
    "explainability",
    "performance_metrics",
)


def _safe(fn: Callable[[], Any]) -> tuple[Any, str | None]:
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}, str(exc)


def _num(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _horizon_metric(forecast: dict[str, Any], key: str, *, prefer_days: int = 5) -> Any:
    """Pull a metric from SqueezeForecast horizons (prefer 5d, else first populated)."""
    horizons = forecast.get("horizons") if isinstance(forecast, dict) else None
    if not isinstance(horizons, list):
        return None
    preferred = next((h for h in horizons if isinstance(h, dict) and h.get("horizon_days") == prefer_days), None)
    ordered = [preferred] + [h for h in horizons if h is not preferred] if preferred else horizons
    for h in ordered:
        if not isinstance(h, dict):
            continue
        if h.get(key) is not None:
            return h.get(key)
    return None


def _extract_findings(symbol: str, results: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact institutional findings row from full stage outputs."""
    squeeze = results.get("gamma_squeeze_engine") or {}
    if isinstance(squeeze, dict) and "data" in squeeze and isinstance(squeeze["data"], dict):
        squeeze = squeeze["data"]
    forecast = squeeze.get("forecast") or {}
    if not isinstance(forecast, dict) or not forecast:
        forecast = squeeze.get("composite") if isinstance(squeeze.get("composite"), dict) else {}
    if not forecast and isinstance(squeeze, dict):
        forecast = squeeze
    meta = forecast.get("meta") if isinstance(forecast.get("meta"), dict) else {}
    composite = meta.get("composite") if isinstance(meta.get("composite"), dict) else {}
    regime = results.get("regime_hmm") or {}
    xgb = results.get("xgboost_direction") or {}
    tft = results.get("tft_forecasting") or {}
    dealer = results.get("dealer_hedging") or {}
    ppo = results.get("ppo_agent") or {}
    hedge = results.get("portfolio_hedging") or {}
    alerts = results.get("trade_alerts") or {}
    explain = results.get("explainability") or {}
    metrics = results.get("performance_metrics") or {}
    patterns = results.get("pattern_recognition") or {}

    alert_list = alerts.get("alerts") or alerts.get("triggered") or []
    if isinstance(alert_list, dict):
        alert_list = alert_list.get("items") or list(alert_list.values())

    flat_metrics = {}
    if isinstance(metrics, dict):
        flat_metrics = metrics.get("flat") or metrics.get("data", {}).get("flat") or {}
        if not flat_metrics and "classification" in metrics:
            for block in ("classification", "regression", "trading", "risk"):
                if isinstance(metrics.get(block), dict):
                    flat_metrics.update(
                        {k: v for k, v in metrics[block].items() if k[:1].isupper()}
                    )

    ok_n = sum(1 for s in stages if s.get("ok"))
    tft_err = isinstance(tft, dict) and bool(tft.get("error"))
    squeeze_err = isinstance(squeeze, dict) and bool(squeeze.get("error"))
    return {
        "symbol": symbol,
        "as_of": (
            forecast.get("as_of")
            or squeeze.get("as_of")
            or regime.get("as_of")
            or dealer.get("as_of")
        ),
        "stages_ok": ok_n,
        "stages_total": len(stages),
        "pipeline_ok": ok_n == len(stages) and not squeeze_err,
        "squeeze_probability": _num(
            forecast.get("probability")
            or forecast.get("p_squeeze")
            or composite.get("probability")
            or squeeze.get("probability")
            or _horizon_metric(forecast, "squeeze_probability")
        ),
        "squeeze_magnitude": _num(
            forecast.get("magnitude")
            or composite.get("magnitude")
            or squeeze.get("magnitude")
            or _horizon_metric(forecast, "expected_magnitude_pct")
        ),
        "squeeze_duration": (
            forecast.get("duration")
            or forecast.get("expected_duration")
            or composite.get("duration")
            or squeeze.get("duration")
            or squeeze.get("expected_duration")
            or _horizon_metric(forecast, "expected_duration_days")
        ),
        "squeeze_confidence": _num(
            forecast.get("confidence")
            or composite.get("confidence")
            or squeeze.get("confidence")
            or _horizon_metric(forecast, "confidence")
        ),
        "squeeze_risk": (
            forecast.get("risk_rating")
            or composite.get("risk_rating")
            or squeeze.get("risk_rating")
            or _horizon_metric(forecast, "risk_rating")
        ),
        "regime": regime.get("current_state") or regime.get("regime") or regime.get("state"),
        "regime_confidence": _num(regime.get("confidence") or regime.get("state_confidence")),
        "xgb_direction": xgb.get("direction") or xgb.get("label") or (xgb.get("summary") or {}).get("direction"),
        "tft_status": "error" if tft_err else "ok",
        "dealer_hedge_volume": _num(
            (dealer.get("hedge_volume") if isinstance(dealer.get("hedge_volume"), (int, float)) else None)
            or (dealer.get("summary") or {}).get("hedge_volume")
            or dealer.get("total_hedge_shares")
        ),
        "dealer_flip": (dealer.get("flip") or dealer.get("summary") or {}).get("flip_level")
        if isinstance(dealer.get("flip") or dealer.get("summary"), dict)
        else dealer.get("flip_level"),
        "ppo_action": ppo.get("action") or ppo.get("structure") or (ppo.get("decision") or {}).get("action"),
        "portfolio_structure": hedge.get("structure")
        or hedge.get("recommended_structure")
        or (hedge.get("recommendation") or {}).get("structure"),
        "optimize_for": hedge.get("optimize_for") or (hedge.get("recommendation") or {}).get("optimize_for"),
        "n_alerts": len(alert_list) if isinstance(alert_list, list) else 0,
        "alert_types": [
            a.get("trigger") or a.get("type") or a.get("name")
            for a in (alert_list if isinstance(alert_list, list) else [])[:12]
            if isinstance(a, dict)
        ],
        "pattern_top": (
            (patterns.get("summary") or {}).get("top_pattern")
            or (patterns.get("top") or [{}])[0].get("name")
            if isinstance(patterns.get("top"), list) and patterns.get("top")
            else (patterns.get("summary") or {}).get("dominant")
        ),
        "explain_has_why": bool(
            explain.get("why")
            or explain.get("decision_trace")
            or (explain.get("explanation") or {}).get("why")
        ),
        "sharpe": _num(flat_metrics.get("Sharpe Ratio")),
        "roc_auc": _num(flat_metrics.get("ROC AUC")),
        "max_drawdown": _num(flat_metrics.get("Maximum Drawdown")),
        "stage_errors": {s["step"]: s["error"] for s in stages if s.get("error")},
        "squeeze_error": squeeze.get("error") if isinstance(squeeze, dict) else None,
    }


def run_symbol_pipeline(
    symbol: str,
    *,
    rebuild_features: bool = False,
    train_models: bool = False,
) -> dict[str, Any]:
    """
    Run the 14-stage product pipeline for one symbol (inference-first by default).

    ``train_models=False`` uses existing artifacts / fit-if-needed light paths so a
    100-name universe scan finishes in institutional batch time.
    """
    from services.data_collection.service import collect_snapshot
    from services.dealer_hedging.service import run_dealer_hedge
    from services.explainability.service import run_explain
    from services.feature_engineering.service import build_or_load_features
    from services.gamma_squeeze_engine.service import run_squeeze
    from services.pattern_recognition.service import run_patterns
    from services.performance_metrics.service import track_symbols
    from services.portfolio_hedging.service import run_portfolio_hedge
    from services.ppo_agent.service import run_ppo
    from services.regime_hmm.service import run_regime
    from services.dashboard.service import build_dashboard
    from services.tft_forecasting.service import run_tft
    from services.trade_alerts.service import evaluate_alerts
    from services.xgboost_direction.service import run_direction

    sym = symbol.upper()
    stages: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    def step(name: str, fn: Callable[[], Any]) -> Any:
        out, err = _safe(fn)
        # Treat payload-level error keys as stage failures (services often return dicts)
        if err is None and isinstance(out, dict) and out.get("error"):
            err = str(out.get("error"))
        results[name] = out
        stages.append(
            {
                "step": name,
                "status": "error" if err else "ok",
                "ok": err is None,
                "error": err,
            }
        )
        return out

    step("data_collection", lambda: collect_snapshot(sym))
    step("feature_engineering", lambda: build_or_load_features(sym, rebuild=rebuild_features))
    step("pattern_recognition", lambda: run_patterns(sym))
    step("regime_hmm", lambda: run_regime(sym, train=train_models))
    step("xgboost_direction", lambda: run_direction(sym, train=train_models))
    step("tft_forecasting", lambda: run_tft(sym, train=train_models))
    step("dealer_hedging", lambda: run_dealer_hedge(sym))
    step("gamma_squeeze_engine", lambda: run_squeeze(sym, persist=True, composite=True))
    step("ppo_agent", lambda: run_ppo(sym, train=False))
    step("portfolio_hedging", lambda: run_portfolio_hedge(sym))
    step("dashboard", lambda: build_dashboard(sym))
    step("trade_alerts", lambda: evaluate_alerts(sym))
    step("explainability", lambda: run_explain(sym, include_forecast=True))
    step("performance_metrics", lambda: track_symbols([sym], rebuild=False))

    findings = _extract_findings(sym, results, stages)
    return {
        "symbol": sym,
        "steps": list(UNIVERSE_PIPELINE_STEPS),
        "stages": stages,
        "findings": findings,
        "results": results,
        "ok": all(s["ok"] for s in stages),
    }


def run_top100_pipeline(
    *,
    top: int = 100,
    refresh_universe: bool = True,
    rebuild_features: bool = False,
    train_models: bool = False,
    limit: int | None = None,
    export_root: Path | None = None,
    persist_full_results: bool = False,
    sync_matrices: bool = True,
) -> dict[str, Any]:
    """
    Run the 14-stage pipeline across top-*n* NASDAQ names by market cap and
    export findings under ``Gamma Squeeze Matrix/scans/phase14_pipeline`` on PortableSSD.
    """
    seed_everything()
    universe = load_top_market_cap(top=top, refresh=refresh_universe)
    if limit:
        universe = universe[:limit]
    symbols = [str(u["symbol"]).upper() for u in universe]
    cap_map = {str(u["symbol"]).upper(): float(u.get("market_cap") or 0) for u in universe}

    matrix_root = resolve_matrix_root()
    export_base = export_root or (matrix_root / "scans" / "phase14_pipeline")
    export_base.mkdir(parents=True, exist_ok=True)
    per_symbol_dir = export_base / "symbols"
    per_symbol_dir.mkdir(parents=True, exist_ok=True)

    # Cache universe on SSD
    uni_path = matrix_root / "top100_nasdaq_by_market_cap.json"
    try:
        uni_path.write_text(json.dumps(universe, indent=2), encoding="utf-8")
    except OSError:
        pass

    matrix_sync: dict[str, Any] = {}
    if sync_matrices:
        from gamma_squeeze.scan.confirmed_squeeze import sync_matrices_to_ssd

        log_event(logger, "universe_pipeline_sync_matrices", n_symbols=len(symbols))
        matrix_sync = sync_matrices_to_ssd(
            symbols,
            dest_root=matrix_root,
            latest_only=True,
            fetch_missing_from_kv=True,
        )
        log_event(
            logger,
            "universe_pipeline_sync_matrices_done",
            kv_fetched=matrix_sync.get("kv_fetched"),
            missing=matrix_sync.get("symbols_missing"),
        )

    # Local export stamp retains time for file uniqueness; KV uploaders use YYYY-MM-DD only.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pipeline_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    findings_rows: list[dict[str, Any]] = []
    stage_fail_counts: dict[str, int] = {s: 0 for s in UNIVERSE_PIPELINE_STEPS}

    for i, sym in enumerate(symbols, 1):
        log_event(
            logger,
            "universe_pipeline_symbol",
            symbol=sym,
            index=i,
            total=len(symbols),
            market_cap=cap_map.get(sym),
        )
        try:
            out = run_symbol_pipeline(
                sym,
                rebuild_features=rebuild_features,
                train_models=train_models,
            )
        except Exception as exc:  # noqa: BLE001
            out = {
                "symbol": sym,
                "ok": False,
                "stages": [],
                "findings": {
                    "symbol": sym,
                    "pipeline_ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-1500:],
                },
                "results": {},
            }
        row = dict(out.get("findings") or {})
        row["market_cap"] = cap_map.get(sym)
        row["rank"] = i
        findings_rows.append(row)
        for s in out.get("stages") or []:
            if not s.get("ok"):
                stage_fail_counts[s["step"]] = stage_fail_counts.get(s["step"], 0) + 1

        # Persist per-symbol summary (optional full payload)
        sym_payload = {
            "symbol": sym,
            "market_cap": cap_map.get(sym),
            "rank": i,
            "findings": row,
            "stages": out.get("stages"),
            "ok": out.get("ok"),
        }
        if persist_full_results:
            sym_payload["results"] = out.get("results")
        with (per_symbol_dir / f"{sym}.json").open("w") as f:
            json.dump(sym_payload, f, indent=2, default=str)

        # Progressive latest snapshot so long runs are inspectable
        if i % 5 == 0 or i == len(symbols):
            _write_export(
                export_base,
                stamp=stamp,
                universe=universe,
                findings_rows=findings_rows,
                stage_fail_counts=stage_fail_counts,
                matrix_root=matrix_root,
                top=top,
                partial=i < len(symbols),
            )

    ranked = sorted(
        findings_rows,
        key=lambda r: (
            float(r.get("squeeze_probability") or 0.0),
            float(r.get("squeeze_confidence") or 0.0),
        ),
        reverse=True,
    )
    top_squeeze = [r for r in ranked if (r.get("squeeze_probability") or 0) >= 0.45][:25]
    with_alerts = [r for r in findings_rows if int(r.get("n_alerts") or 0) > 0]
    pipeline_ok = [r for r in findings_rows if r.get("pipeline_ok")]

    payload = _write_export(
        export_base,
        stamp=stamp,
        universe=universe,
        findings_rows=findings_rows,
        stage_fail_counts=stage_fail_counts,
        matrix_root=matrix_root,
        top=top,
        partial=False,
        extras={
            "top_squeeze_candidates": top_squeeze,
            "with_alerts": with_alerts,
            "n_pipeline_ok": len(pipeline_ok),
            "ranked_by_squeeze_probability": ranked[:50],
            "pipeline_date": pipeline_day,
            "matrix_sync": matrix_sync,
            "kv_key_format": {
                "scan": f"scans/phase14_pipeline/{pipeline_day}",
                "symbol_pipeline": f"{{TICKER}}/pipeline/{pipeline_day}",
                "symbol_forecast": f"{{TICKER}}/forecast/{pipeline_day}",
                "note": "YYYY-MM-DD only — no T153655Z / T061106Z suffixes",
            },
        },
    )
    return payload


def _write_export(
    export_base: Path,
    *,
    stamp: str,
    universe: list[dict[str, Any]],
    findings_rows: list[dict[str, Any]],
    stage_fail_counts: dict[str, int],
    matrix_root: Path,
    top: int,
    partial: bool,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = sorted(
        findings_rows,
        key=lambda r: float(r.get("squeeze_probability") or 0.0),
        reverse=True,
    )
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_id": stamp,
        "partial": partial,
        "universe": "top_nasdaq_by_market_cap",
        "top_n": top,
        "n_scanned": len(findings_rows),
        "pipeline_steps": list(UNIVERSE_PIPELINE_STEPS),
        "n_pipeline_steps": len(UNIVERSE_PIPELINE_STEPS),
        "matrix_root": str(matrix_root),
        "ssd_mounted": Path("/Volumes/PortableSSD").is_dir(),
        "export_dir": str(export_base),
        "stage_fail_counts": stage_fail_counts,
        "n_pipeline_ok": sum(1 for r in findings_rows if r.get("pipeline_ok")),
        "top_squeeze_candidates": [
            r for r in ranked if (r.get("squeeze_probability") or 0) >= 0.45
        ][:25],
        "ranked_by_squeeze_probability": ranked[:50],
        "with_alerts": [r for r in findings_rows if int(r.get("n_alerts") or 0) > 0],
        "findings": findings_rows,
        "universe_tickers": universe,
    }
    if extras:
        payload.update(extras)

    latest = export_base / "latest.json"
    stamped = export_base / f"pipeline_{stamp}.json"
    with latest.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    with stamped.open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    # CSV summary for desk use
    csv_path = export_base / f"findings_{stamp}.csv"
    csv_latest = export_base / "findings_latest.csv"
    fields = [
        "rank",
        "symbol",
        "market_cap",
        "as_of",
        "pipeline_ok",
        "stages_ok",
        "squeeze_probability",
        "squeeze_magnitude",
        "squeeze_duration",
        "squeeze_confidence",
        "squeeze_risk",
        "regime",
        "regime_confidence",
        "xgb_direction",
        "ppo_action",
        "portfolio_structure",
        "n_alerts",
        "pattern_top",
        "sharpe",
        "roc_auc",
        "max_drawdown",
        "explain_has_why",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in findings_rows:
            w.writerow(row)
    with csv_latest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in findings_rows:
            w.writerow(row)

    with (export_base / "index.json").open("w") as f:
        json.dump(
            {
                "latest": str(latest),
                "last_pipeline": str(stamped),
                "last_csv": str(csv_path),
                "findings_csv": str(csv_latest),
                "n_scanned": len(findings_rows),
                "n_pipeline_ok": payload["n_pipeline_ok"],
                "generated_at": payload["generated_at"],
                "partial": partial,
            },
            f,
            indent=2,
        )
    return payload
