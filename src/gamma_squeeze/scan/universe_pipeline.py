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


def _extract_findings(symbol: str, results: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact institutional findings row from full stage outputs."""
    squeeze = results.get("gamma_squeeze_engine") or {}
    if isinstance(squeeze, dict) and "data" in squeeze and isinstance(squeeze["data"], dict):
        squeeze = squeeze["data"]
    forecast = squeeze.get("forecast") or squeeze.get("composite") or squeeze
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
    scores = squeeze.get("scores") if isinstance(squeeze.get("scores"), dict) else {}
    horizons = forecast.get("horizons") if isinstance(forecast.get("horizons"), list) else []
    h5 = next((h for h in horizons if isinstance(h, dict) and h.get("horizon_days") == 5), None)
    if h5 is None and horizons and isinstance(horizons[0], dict):
        h5 = horizons[min(4, len(horizons) - 1)]
    return {
        "symbol": symbol,
        "as_of": squeeze.get("as_of") or regime.get("as_of") or dealer.get("as_of"),
        "stages_ok": ok_n,
        "stages_total": len(stages),
        "pipeline_ok": ok_n == len(stages),
        "squeeze_probability": _num(
            squeeze.get("probability")
            or forecast.get("probability")
            or forecast.get("p_squeeze")
            or (h5 or {}).get("squeeze_probability")
            or scores.get("gamma_squeeze_score")
        ),
        "squeeze_magnitude": _num(
            squeeze.get("magnitude")
            or forecast.get("magnitude")
            or (h5 or {}).get("expected_magnitude_pct")
        ),
        "squeeze_duration": (
            squeeze.get("duration")
            or forecast.get("duration")
            or (h5 or {}).get("expected_duration_days")
        ),
        "squeeze_confidence": _num(
            squeeze.get("confidence")
            or forecast.get("confidence")
            or (h5 or {}).get("confidence_score")
        ),
        "squeeze_risk": (
            squeeze.get("risk_rating")
            or forecast.get("risk_rating")
            or (h5 or {}).get("risk_rating")
        ),
        "regime": regime.get("current_state") or regime.get("regime") or regime.get("state"),
        "regime_confidence": _num(regime.get("confidence") or regime.get("state_confidence")),
        "xgb_direction": xgb.get("direction") or xgb.get("label") or (xgb.get("summary") or {}).get("direction"),
        "tft_status": "ok" if "error" not in tft else "error",
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
    matrix_lookback_dates: int = 10,
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

        # Pull recent dated matrices from alpaca-options-matrix-backup when SSD empty.
        # latest_only=False fetches up to 5 KV dates; extra lookback via repeated sync
        # is handled inside sync (kv_dates[-5:]). Callers may raise lookback later.
        latest_only = matrix_lookback_dates <= 1
        log_event(
            logger,
            "universe_pipeline_matrix_sync_start",
            n_symbols=len(symbols),
            latest_only=latest_only,
            lookback=matrix_lookback_dates,
        )
        matrix_sync = sync_matrices_to_ssd(
            symbols,
            dest_root=matrix_root,
            latest_only=latest_only,
        )
        # If caller asked for more than 5, fetch additional trailing dates directly.
        if matrix_lookback_dates > 5:
            try:
                from gamma_squeeze.ingest.alpaca_backup_client import (
                    fetch_options_matrix,
                    list_matrix_dates,
                )

                extra_fetched = 0
                for sym in symbols:
                    kv_dates = list_matrix_dates(sym)
                    if not kv_dates:
                        continue
                    use = kv_dates[-matrix_lookback_dates:]
                    out_dir = matrix_root / "tickers" / sym
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for d in use:
                        path = out_dir / f"{d}.json"
                        if path.is_file():
                            continue
                        matrix = fetch_options_matrix(sym, d)
                        path.write_text(json.dumps(matrix), encoding="utf-8")
                        extra_fetched += 1
                matrix_sync["extra_lookback_fetched"] = extra_fetched
                matrix_sync["matrix_lookback_dates"] = matrix_lookback_dates
            except Exception as exc:  # noqa: BLE001
                matrix_sync["extra_lookback_error"] = str(exc)
        log_event(
            logger,
            "universe_pipeline_matrix_sync_done",
            **{k: matrix_sync.get(k) for k in ("kv_fetched", "copied_files", "symbols_missing")},
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
            "matrix_sync": matrix_sync,
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
