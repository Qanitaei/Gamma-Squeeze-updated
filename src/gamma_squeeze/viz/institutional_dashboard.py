"""Institutional Visualization Dashboard — panels + interactive heatmaps.

Data plane: Alpaca API (options matrix / OHLCV), then SSD / features fallback.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from gamma_squeeze.config import alpaca_configured, resolve_matrix_root
from gamma_squeeze.features.feature_store import build_features_for_symbol, load_features
from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix
from gamma_squeeze.ingest.macro_client import fetch_calendar_near
from gamma_squeeze.serve.export_forecasts import load_latest_forecast


DASHBOARD_PANELS: tuple[str, ...] = (
    "Market Regime",
    "Dealer Positioning",
    "Gamma Exposure",
    "Net GEX",
    "Gamma Flip",
    "Call Wall",
    "Put Wall",
    "Forecast",
    "Probability",
    "Expected Move",
    "Pattern Detection",
    "Portfolio",
    "Greeks",
    "Risk",
    "Macro Dashboard",
    "Economic Calendar",
    "Alert Center",
    "Trade Journal",
    "Performance Analytics",
)

HEATMAPS: tuple[str, ...] = (
    "Options Surface",
    "3D Gamma Surface",
    "Volatility Surface",
    "Dealer Position Map",
    "Liquidity Map",
    "Strike Distribution",
)


def _f(row: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return default


def _contracts(matrix: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not matrix:
        return []
    if matrix.get("contracts"):
        return list(matrix["contracts"])
    rows: list[dict[str, Any]] = []
    for exp in matrix.get("by_expiration") or []:
        for strike_block in exp.get("strikes") or []:
            for side in ("call", "put"):
                c = strike_block.get(side)
                if c:
                    rows.append(c)
    return rows


def _load_alpaca_matrix(symbol: str, as_of: str | None) -> tuple[dict[str, Any] | None, str]:
    """Prefer live Alpaca API, then SSD."""
    if alpaca_configured():
        try:
            from gamma_squeeze.ingest.alpaca_api import fetch_live_options_matrix

            live = fetch_live_options_matrix(symbol, as_of=as_of)
            if live.get("contracts"):
                return live, "alpaca_api"
        except Exception:  # noqa: BLE001
            pass
    root = resolve_matrix_root()
    if as_of:
        local = load_local_matrix(root, symbol, as_of)
        if local:
            return local, "ssd"
    dates = list_local_dates(root, symbol) or []
    if dates:
        local = load_local_matrix(root, symbol, dates[-1])
        if local:
            return local, "ssd"
    return None, "none"


def _panel_market_regime(row: dict[str, Any], regime: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "title": "Market Regime",
        "current_state": (regime or {}).get("current_state") or row.get("regime") or (
            "Negative Gamma" if _f(row, "regime_neg_gamma", default=0) >= 0.5 else "Neutral"
        ),
        "confidence": (regime or {}).get("confidence") or _f(row, "regime_confidence", default=0.5),
        "neg_gamma": _f(row, "regime_neg_gamma", default=0.0),
        "positioning_stress": _f(row, "positioning_stress", default=0.0),
        "rvol_10d": _f(row, "rvol_10d", default=None),
    }


def _panel_dealer(row: dict[str, Any], dealer: dict[str, Any] | None) -> dict[str, Any]:
    d = dealer or {}
    return {
        "title": "Dealer Positioning",
        "delta": _f(row, "dealer_delta", default=d.get("delta_hedging")),
        "gamma": _f(row, "dealer_gamma"),
        "hedge_requirement": _f(row, "dealer_hedge_requirement"),
        "exhaustion": _f(row, "dealer_exhaustion", default=d.get("dealer_exhaustion")),
        "liquidity": _f(row, "dealer_liquidity_score", default=d.get("dealer_liquidity")),
        "inventory": d.get("dealer_inventory"),
        "share_purchases": d.get("dealer_share_purchases") or d.get("share_purchases"),
        "share_sales": d.get("dealer_share_sales") or d.get("share_sales"),
    }


def _panel_gex(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "Gamma Exposure",
        "net_gex": _f(row, "net_gex", "gex_per_spot"),
        "gamma_exposure": _f(row, "gamma_exposure"),
        "gex_per_spot": _f(row, "gex_per_spot"),
        "regime": "negative" if (_f(row, "net_gex", default=0) or 0) < 0 else "positive",
    }


def _panel_levels(row: dict[str, Any]) -> dict[str, Any]:
    spot = _f(row, "spot", default=0) or 0
    flip = _f(row, "gamma_flip", "zero_gamma")
    cw = _f(row, "call_wall")
    pw = _f(row, "put_wall")
    return {
        "gamma_flip": {"title": "Gamma Flip", "level": flip, "distance_pct": ((flip - spot) / spot * 100) if flip and spot else None},
        "call_wall": {"title": "Call Wall", "level": cw, "distance_pct": ((cw - spot) / spot * 100) if cw and spot else None},
        "put_wall": {"title": "Put Wall", "level": pw, "distance_pct": ((pw - spot) / spot * 100) if pw and spot else None},
        "spot": spot,
    }


def _panel_forecast(forecast: dict[str, Any] | None, composite: dict[str, Any] | None) -> dict[str, Any]:
    horizons = (forecast or {}).get("horizons") or []
    h5 = next((h for h in horizons if h.get("horizon_days") == 5), horizons[0] if horizons else {})
    comp = composite or (forecast or {}).get("meta", {}).get("composite") or {}
    return {
        "title": "Forecast",
        "probability": comp.get("probability", h5.get("squeeze_probability")),
        "expected_move": comp.get("magnitude", h5.get("expected_magnitude_pct")),
        "expected_duration": comp.get("expected_duration", h5.get("expected_duration_days")),
        "expected_start": comp.get("expected_start"),
        "confidence": comp.get("confidence", h5.get("confidence_score")),
        "risk_rating": comp.get("risk_rating"),
        "term_structure": [
            {
                "horizon_days": h.get("horizon_days"),
                "probability": h.get("squeeze_probability"),
                "expected_move": h.get("expected_magnitude_pct"),
                "confidence": h.get("confidence_score"),
            }
            for h in horizons
        ],
        "why": ((forecast or {}).get("meta") or {}).get("why"),
    }


def _panel_patterns(patterns: dict[str, Any] | None) -> dict[str, Any]:
    pats = (patterns or {}).get("patterns") or (patterns or {}).get("top") or []
    return {
        "title": "Pattern Detection",
        "n_patterns": len(pats),
        "top": pats[:8],
        "compression": (patterns or {}).get("compression")
        or ((patterns or {}).get("summary") or {}).get("compression"),
    }


def _panel_portfolio_greeks_risk(row: dict[str, Any], hedges: dict[str, Any] | None) -> dict[str, Any]:
    book = (hedges or {}).get("book") or {}
    return {
        "portfolio": {
            "title": "Portfolio",
            "optimized": (hedges or {}).get("optimized") or {},
            "top_structures": [
                r.get("structure")
                for r in ((hedges or {}).get("recommendations") or [])[:5]
            ],
            "book": book,
        },
        "greeks": {
            "title": "Greeks",
            "delta": _f(row, "dealer_delta", default=book.get("delta")),
            "gamma": _f(row, "dealer_gamma", default=book.get("gamma")),
            "vega": _f(row, "dealer_vega", default=book.get("vega")),
            "theta": _f(row, "dealer_theta", default=book.get("theta")),
            "charm": _f(row, "dealer_charm"),
            "vanna": _f(row, "dealer_vanna"),
        },
        "risk": {
            "title": "Risk",
            "positioning_stress": _f(row, "positioning_stress"),
            "rvol_10d": _f(row, "rvol_10d"),
            "atm_iv": _f(row, "atm_iv"),
            "risk_rating": ((hedges or {}).get("meta") or {}).get("signals", {}).get("confidence"),
            "dealer_exhaustion": _f(row, "dealer_exhaustion"),
        },
    }


def _panel_macro(row: dict[str, Any], as_of: str) -> dict[str, Any]:
    cal: list[dict[str, Any]] = []
    try:
        cal = fetch_calendar_near(as_of or "", look_ahead_days=14)
    except Exception:  # noqa: BLE001
        cal = []
    return {
        "macro_dashboard": {
            "title": "Macro Dashboard",
            "vix": _f(row, "vix"),
            "fed_funds": _f(row, "fed_funds"),
            "yield_spread": _f(row, "yield_spread"),
            "credit_spread": _f(row, "credit_spread"),
            "consumer_sentiment": _f(row, "consumer_sentiment"),
            "dollar_index": _f(row, "dollar_index"),
        },
        "economic_calendar": {
            "title": "Economic Calendar",
            "events": cal[:25],
            "n_events": len(cal),
        },
    }


def _panel_alerts_journal_perf(
    forecast: dict[str, Any] | None,
    alerts: dict[str, Any] | None,
    features: pd.DataFrame,
) -> dict[str, Any]:
    horizons = (forecast or {}).get("horizons") or []
    journal = []
    for h in horizons[:5]:
        journal.append(
            {
                "as_of": (forecast or {}).get("as_of"),
                "action": "watch" if float(h.get("squeeze_probability") or 0) >= 0.4 else "stand_aside",
                "horizon_days": h.get("horizon_days"),
                "probability": h.get("squeeze_probability"),
                "expected_move": h.get("expected_magnitude_pct"),
                "note": "auto-journal from forecast term structure",
            }
        )
    # Performance: rolling feature-based proxy
    rets = []
    if not features.empty and "ret_1d" in features.columns:
        rets = pd.to_numeric(features["ret_1d"], errors="coerce").dropna().tail(60).tolist()
    arr = np.asarray(rets, dtype=float)
    sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(252)) if len(arr) > 2 else 0.0
    return {
        "alert_center": {
            "title": "Alert Center",
            "fired": bool((alerts or {}).get("fired")),
            "alerts": (alerts or {}).get("alerts") or [],
            "thresholds": (alerts or {}).get("thresholds") or {},
        },
        "trade_journal": {"title": "Trade Journal", "entries": journal},
        "performance_analytics": {
            "title": "Performance Analytics",
            "n_bars": int(len(arr)),
            "realized_vol": float(np.std(arr) * np.sqrt(252)) if len(arr) > 1 else None,
            "sharpe_proxy": sharpe,
            "mean_ret_1d": float(np.mean(arr)) if len(arr) else None,
            "hit_rate_proxy": float(np.mean(np.asarray(arr) > 0)) if len(arr) else None,
        },
    }


def _surface_grids(contracts: list[dict[str, Any]], spot: float) -> dict[str, Any]:
    """Build strike × DTE grids for IV / gamma / OI / liquidity."""
    if not contracts:
        return {
            "strikes": [],
            "dtes": [],
            "iv": [],
            "gamma": [],
            "oi": [],
            "volume": [],
            "dealer_sign": [],
        }

    def _strike(c):
        return float(c.get("strike_price") or c.get("strike") or 0)

    def _dte(c):
        v = c.get("days_to_expiration")
        if v is not None:
            return int(float(v))
        return None

    def _iv(c):
        return float(c.get("implied_volatility") or c.get("volatility") or c.get("iv") or 0)

    def _gamma(c):
        return float(c.get("gamma") or 0)

    def _oi(c):
        return float(c.get("open_interest") or c.get("oi") or 0)

    def _vol(c):
        return float(c.get("total_volume") or c.get("volume") or 0)

    def _side(c):
        pc = str(c.get("put_call") or c.get("type") or "").lower()
        if pc.startswith("c"):
            return 1.0
        if pc.startswith("p"):
            return -1.0
        return 0.0

    # Filter near spot
    near = [c for c in contracts if _strike(c) > 0 and (_dte(c) is not None and 0 <= (_dte(c) or 0) <= 90)]
    if not near:
        near = [c for c in contracts if _strike(c) > 0][:500]
    if spot > 0:
        near = [c for c in near if abs(_strike(c) / spot - 1.0) <= 0.15] or near

    strikes = sorted({round(_strike(c), 4) for c in near})
    dtes = sorted({int(_dte(c) or 0) for c in near})
    # downsample grids for payload size
    if len(strikes) > 40:
        idx = np.linspace(0, len(strikes) - 1, 40).astype(int)
        strikes = [strikes[i] for i in idx]
    if len(dtes) > 12:
        idx = np.linspace(0, len(dtes) - 1, 12).astype(int)
        dtes = [dtes[i] for i in idx]

    strike_set = set(strikes)
    dte_set = set(dtes)
    buckets: dict[tuple[float, int], list[dict[str, float]]] = defaultdict(list)
    for c in near:
        s, d = round(_strike(c), 4), int(_dte(c) or 0)
        if s not in strike_set or d not in dte_set:
            continue
        buckets[(s, d)].append(
            {
                "iv": _iv(c),
                "gamma": _gamma(c) * _oi(c) * 100 * (spot**2) * 0.01 * _side(c),  # rough GEX sign
                "oi": _oi(c),
                "volume": _vol(c),
                "side": _side(c),
            }
        )

    def grid(key: str) -> list[list[float | None]]:
        out: list[list[float | None]] = []
        for d in dtes:
            row = []
            for s in strikes:
                vals = buckets.get((s, d)) or []
                if not vals:
                    row.append(None)
                else:
                    row.append(float(np.mean([v[key] for v in vals])))
            out.append(row)
        return out

    # Liquidity = 0.6*oi_norm + 0.4*vol_norm per cell
    oi_g = grid("oi")
    vol_g = grid("volume")
    flat_oi = [x for row in oi_g for x in row if x is not None]
    flat_vol = [x for row in vol_g for x in row if x is not None]
    max_oi = max(flat_oi) if flat_oi else 0.0
    max_vol = max(flat_vol) if flat_vol else 0.0
    if max_oi <= 0:
        max_oi = 1.0
    if max_vol <= 0:
        max_vol = 1.0
    liq = []
    for i, d in enumerate(dtes):
        row = []
        for j, s in enumerate(strikes):
            oi = oi_g[i][j]
            vv = vol_g[i][j]
            if oi is None and vv is None:
                row.append(None)
            else:
                row.append(0.6 * ((oi or 0) / max_oi) + 0.4 * ((vv or 0) / max_vol))
        liq.append(row)

    # Strike distribution (sum OI by strike)
    oi_by_strike: dict[float, float] = defaultdict(float)
    call_oi: dict[float, float] = defaultdict(float)
    put_oi: dict[float, float] = defaultdict(float)
    for c in near:
        s = round(_strike(c), 4)
        oi = _oi(c)
        oi_by_strike[s] += oi
        if _side(c) > 0:
            call_oi[s] += oi
        elif _side(c) < 0:
            put_oi[s] += oi

    strike_dist = {
        "strikes": sorted(oi_by_strike.keys()),
        "total_oi": [oi_by_strike[s] for s in sorted(oi_by_strike.keys())],
        "call_oi": [call_oi[s] for s in sorted(oi_by_strike.keys())],
        "put_oi": [put_oi[s] for s in sorted(oi_by_strike.keys())],
    }

    return {
        "strikes": strikes,
        "dtes": dtes,
        "iv": grid("iv"),
        "gamma": grid("gamma"),
        "oi": oi_g,
        "volume": vol_g,
        "liquidity": liq,
        "dealer_gex": grid("gamma"),
        "strike_distribution": strike_dist,
        "spot": spot,
    }


def _plotly_specs(surfaces: dict[str, Any]) -> dict[str, Any]:
    strikes = surfaces.get("strikes") or []
    dtes = surfaces.get("dtes") or []
    return {
        "volatility_surface": {
            "type": "surface",
            "x": strikes,
            "y": dtes,
            "z": surfaces.get("iv") or [],
            "title": "Volatility Surface",
        },
        "gamma_surface_3d": {
            "type": "surface",
            "x": strikes,
            "y": dtes,
            "z": surfaces.get("gamma") or [],
            "title": "3D Gamma Surface",
        },
        "options_surface": {
            "type": "heatmap",
            "x": strikes,
            "y": dtes,
            "z": surfaces.get("oi") or [],
            "title": "Options Surface (OI)",
        },
        "dealer_position_map": {
            "type": "heatmap",
            "x": strikes,
            "y": dtes,
            "z": surfaces.get("dealer_gex") or [],
            "title": "Dealer Position Map (signed GEX)",
        },
        "liquidity_map": {
            "type": "heatmap",
            "x": strikes,
            "y": dtes,
            "z": surfaces.get("liquidity") or [],
            "title": "Liquidity Map",
        },
        "strike_distribution": {
            "type": "bar",
            "x": (surfaces.get("strike_distribution") or {}).get("strikes") or [],
            "y": (surfaces.get("strike_distribution") or {}).get("total_oi") or [],
            "title": "Strike Distribution",
        },
    }


def build_institutional_dashboard(
    symbol: str,
    *,
    use_live_alpaca: bool = True,
    include_pipeline_extras: bool = False,
) -> dict[str, Any]:
    """Assemble full institutional dashboard payload for one symbol.

    Set ``include_pipeline_extras=True`` to pull regime/patterns/hedges/alerts/dealer
    via in-process service calls. Surfaces always prefer Alpaca matrix data.
    """
    sym = symbol.upper()
    feat = load_features(sym)
    if feat.empty:
        feat = build_features_for_symbol(
            sym,
            include_skew=False,
            include_macro=True,
            include_technicals=True,
            include_dealer=True,
            include_options_metrics=True,
        )
    if feat.empty:
        return {
            "symbol": sym,
            "error": "no_features",
            "panels": {},
            "heatmaps": {},
            "data_origin": "none",
        }

    row = feat.sort_values("as_of").iloc[-1].to_dict()
    as_of = str(row.get("as_of"))[:10]

    matrix = None
    origin = "features"
    if use_live_alpaca:
        matrix, origin = _load_alpaca_matrix(sym, as_of)

    forecast = load_latest_forecast(sym)
    composite = None
    regime = None
    patterns = None
    hedges = None
    alerts = None
    dealer = None

    if include_pipeline_extras:
        # Best-effort enrichment (library imports; never Schwab)
        try:
            from services.gamma_squeeze_engine.service import run_squeeze

            if not forecast:
                sq = run_squeeze(sym, persist=True, composite=True)
                forecast = sq.get("forecast")
                composite = {
                    "probability": sq.get("probability"),
                    "magnitude": sq.get("magnitude"),
                    "expected_duration": sq.get("expected_duration"),
                    "expected_start": sq.get("expected_start"),
                    "confidence": sq.get("confidence"),
                    "risk_rating": sq.get("risk_rating"),
                }
                if not as_of:
                    as_of = sq.get("as_of") or as_of
        except Exception:  # noqa: BLE001
            pass
        try:
            from services.regime_hmm.service import run_regime

            regime = run_regime(sym, train=False)
        except Exception:  # noqa: BLE001
            regime = None
        try:
            from services.pattern_recognition.service import run_patterns

            patterns = run_patterns(sym)
        except Exception:  # noqa: BLE001
            patterns = None
        try:
            from services.portfolio_hedging.service import run_portfolio_hedge

            hedges = run_portfolio_hedge(sym, use_live_alpaca=use_live_alpaca)
        except Exception:  # noqa: BLE001
            hedges = None
        try:
            from services.trade_alerts.service import evaluate_alerts

            alerts = evaluate_alerts(sym)
        except Exception:  # noqa: BLE001
            alerts = None
        try:
            from services.dealer_hedging.service import run_dealer_hedge

            dealer = run_dealer_hedge(sym, as_of=as_of)
        except Exception:  # noqa: BLE001
            dealer = None

    levels = _panel_levels(row)
    pgr = _panel_portfolio_greeks_risk(row, hedges)
    macro = _panel_macro(row, as_of)
    ajp = _panel_alerts_journal_perf(forecast, alerts, feat)
    forecast_panel = _panel_forecast(forecast, composite)

    panels = {
        "market_regime": _panel_market_regime(row, regime),
        "dealer_positioning": _panel_dealer(row, dealer),
        "gamma_exposure": _panel_gex(row),
        "net_gex": {"title": "Net GEX", "value": _f(row, "net_gex", "gex_per_spot"), "regime": _panel_gex(row)["regime"]},
        "gamma_flip": levels["gamma_flip"],
        "call_wall": levels["call_wall"],
        "put_wall": levels["put_wall"],
        "levels": levels,
        "forecast": forecast_panel,
        "probability": {
            "title": "Probability",
            "value": forecast_panel.get("probability"),
            "term_structure": forecast_panel.get("term_structure"),
        },
        "expected_move": {
            "title": "Expected Move",
            "value": forecast_panel.get("expected_move"),
            "duration": forecast_panel.get("expected_duration"),
        },
        "pattern_detection": _panel_patterns(patterns),
        "portfolio": pgr["portfolio"],
        "greeks": pgr["greeks"],
        "risk": pgr["risk"],
        "macro_dashboard": macro["macro_dashboard"],
        "economic_calendar": macro["economic_calendar"],
        "alert_center": ajp["alert_center"],
        "trade_journal": ajp["trade_journal"],
        "performance_analytics": ajp["performance_analytics"],
    }

    spot = float(levels.get("spot") or 0)
    if matrix and not spot:
        spot = float(matrix.get("underlying_price") or matrix.get("spot") or 0)
    surfaces = _surface_grids(_contracts(matrix), spot)
    plotly = _plotly_specs(surfaces)
    heatmaps = {
        "options_surface": {"title": "Options Surface", **{k: surfaces.get(k) for k in ("strikes", "dtes", "oi")}},
        "gamma_surface_3d": {"title": "3D Gamma Surface", **{k: surfaces.get(k) for k in ("strikes", "dtes", "gamma")}},
        "volatility_surface": {"title": "Volatility Surface", **{k: surfaces.get(k) for k in ("strikes", "dtes", "iv")}},
        "dealer_position_map": {
            "title": "Dealer Position Map",
            **{k: surfaces.get(k) for k in ("strikes", "dtes", "dealer_gex")},
        },
        "liquidity_map": {"title": "Liquidity Map", **{k: surfaces.get(k) for k in ("strikes", "dtes", "liquidity")}},
        "strike_distribution": {"title": "Strike Distribution", **(surfaces.get("strike_distribution") or {})},
    }

    # Alpaca account/health snapshot (replaces local meta curl pattern)
    alpaca_health: dict[str, Any] = {"configured": alpaca_configured()}
    if alpaca_configured():
        try:
            from gamma_squeeze.ingest.alpaca_api import alpaca_api_health

            alpaca_health = alpaca_api_health()
        except Exception as exc:  # noqa: BLE001
            alpaca_health = {"configured": True, "ok": False, "error": str(exc)}

    return {
        "symbol": sym,
        "as_of": as_of,
        "data_origin": origin,
        "alpaca": alpaca_health,
        "panels": panels,
        "heatmaps": heatmaps,
        "plotly": plotly,
        "catalog": {"panels": list(DASHBOARD_PANELS), "heatmaps": list(HEATMAPS)},
        "why": forecast_panel.get("why"),
        "explanations": (forecast or {}).get("explanations") or [],
        "explainability": ((forecast or {}).get("meta") or {}).get("explainability") or {},
    }



def dashboard_meta() -> dict[str, Any]:
    """Dashboard meta via Alpaca API health (no local-only curl stub)."""
    from gamma_squeeze.ingest.alpaca_api import alpaca_api_health

    return {
        "service": "dashboard",
        "panels": list(DASHBOARD_PANELS),
        "heatmaps": list(HEATMAPS),
        "data_plane": "alpaca",
        "alpaca": alpaca_api_health(),
    }
