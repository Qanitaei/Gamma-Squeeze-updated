"""Locate and test confirmed gamma squeezes across a market-cap universe."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gamma_squeeze.config import FALLBACK_ROOT, resolve_matrix_root
from gamma_squeeze.core.logging import get_logger, log_event
from gamma_squeeze.core.reproducibility import seed_everything
from gamma_squeeze.features.gex_features import gex_feature_row
from gamma_squeeze.features.options_metrics import compute_options_metrics
from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix
from gamma_squeeze.ingest.nasdaq_universe import top_nasdaq_by_market_cap
from gamma_squeeze.models.composite_squeeze import BREAKOUT_PATTERNS, SQUEEZE_REGIMES
from gamma_squeeze.patterns.recognition import detect_patterns

logger = get_logger("scan.confirmed_squeeze")

CONFIRMED_THRESHOLD = 0.58

DEFAULT_TOP100_CACHE = FALLBACK_ROOT / "top100_nasdaq_by_market_cap.json"
DEFAULT_ALPACA_MATRIX_STORE = FALLBACK_ROOT / "tickers"


@dataclass
class ConfirmedSqueezeHit:
    symbol: str
    as_of: str
    confirmed: bool
    confirmation_score: float
    reasons: list[str] = field(default_factory=list)
    market_cap: float | None = None
    gex: dict[str, Any] = field(default_factory=dict)
    dex: dict[str, Any] = field(default_factory=dict)
    greeks: dict[str, Any] = field(default_factory=dict)
    options_metrics: dict[str, Any] = field(default_factory=dict)
    patterns: dict[str, Any] = field(default_factory=dict)
    breakouts: list[dict[str, Any]] = field(default_factory=list)
    candlesticks: list[dict[str, Any]] = field(default_factory=list)
    data_sources: dict[str, Any] = field(default_factory=dict)
    composite: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_top100_cache(cache_path: Path | None = None) -> Path:
    if cache_path is not None:
        return cache_path
    candidates = [
        DEFAULT_TOP100_CACHE,
        resolve_matrix_root() / "top100_nasdaq_by_market_cap.json",
        Path("/Volumes/PortableSSD/Gamma Squeeze Matrix/top100_nasdaq_by_market_cap.json"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return DEFAULT_TOP100_CACHE


def load_top_market_cap(
    *,
    top: int = 100,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Top NASDAQ names by market cap (cached JSON or live Nasdaq screener)."""
    path = _resolve_top100_cache(cache_path)
    if path.is_file() and not refresh:
        rows = json.loads(path.read_text())
        if isinstance(rows, list) and rows:
            return rows[:top]
    # Force live screener fetch when refreshing (bypass stale cache).
    if refresh and path.is_file():
        try:
            path.unlink()
        except OSError:
            pass
    rows = top_nasdaq_by_market_cap(None, top, cache_path=path)
    # Mirror onto PortableSSD matrix root when available.
    ssd_cache = resolve_matrix_root() / "top100_nasdaq_by_market_cap.json"
    if ssd_cache != path:
        try:
            ssd_cache.parent.mkdir(parents=True, exist_ok=True)
            ssd_cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        except OSError:
            pass
    return rows[:top]


def sync_matrices_to_ssd(
    symbols: list[str],
    *,
    source_root: Path | None = None,
    dest_root: Path | None = None,
    latest_only: bool = True,
    lookback_dates: int | None = None,
    fetch_missing_from_kv: bool = True,
) -> dict[str, Any]:
    """Copy Alpaca historical matrices into Gamma Squeeze Matrix tickers/ on SSD (or fallback).

    When a symbol is absent from the local store, optionally pull dated matrices
    from the alpaca-options-matrix-backup worker. ``lookback_dates`` (e.g. 30)
    fetches the newest N trading days from KV even when a local copy exists.
    """
    src = source_root or DEFAULT_ALPACA_MATRIX_STORE
    dest = dest_root or resolve_matrix_root()
    n_lookback = int(lookback_dates) if lookback_dates and lookback_dates > 0 else None
    copied = 0
    missing = 0
    kv_fetched = 0
    skipped_same = 0
    kv_errors: list[str] = []
    for sym in symbols:
        sdir = src / sym.upper()
        dates = sorted(p.stem for p in sdir.glob("*.json")) if sdir.is_dir() else []
        out_dir = dest / "tickers" / sym.upper()
        if dates and n_lookback is None:
            use = dates[-1:] if latest_only else dates
            out_dir.mkdir(parents=True, exist_ok=True)
            for d in use:
                src_path = sdir / f"{d}.json"
                dst_path = out_dir / f"{d}.json"
                try:
                    if src_path.resolve() == dst_path.resolve():
                        skipped_same += 1
                        continue
                except OSError:
                    pass
                try:
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                except shutil.SameFileError:
                    skipped_same += 1
            # Without an explicit lookback, local copies are sufficient.
            continue

        if not fetch_missing_from_kv:
            if not list_local_dates(dest, sym) and not dates:
                missing += 1
            continue
        try:
            from gamma_squeeze.ingest.alpaca_backup_client import (
                fetch_options_matrix,
                list_matrix_dates,
            )

            kv_dates = list_matrix_dates(sym)
            if not kv_dates:
                if not list_local_dates(dest, sym) and not dates:
                    missing += 1
                continue
            if n_lookback is not None:
                use = kv_dates[-n_lookback:]
            elif latest_only:
                use = kv_dates[-1:]
            else:
                use = kv_dates[-5:]
            out_dir.mkdir(parents=True, exist_ok=True)
            for d in use:
                dst_path = out_dir / f"{d}.json"
                if dst_path.is_file() and dst_path.stat().st_size > 50:
                    continue
                matrix = fetch_options_matrix(sym, d)
                with dst_path.open("w") as f:
                    json.dump(matrix, f)
                kv_fetched += 1
        except Exception as exc:  # noqa: BLE001
            if not list_local_dates(dest, sym):
                missing += 1
            kv_errors.append(f"{sym}: {exc}")
    return {
        "source": str(src),
        "dest": str(dest),
        "copied_files": copied,
        "kv_fetched": kv_fetched,
        "skipped_same_file": skipped_same,
        "lookback_dates": n_lookback,
        "symbols_missing": missing,
        "kv_errors": kv_errors[:20],
        "ssd_mounted": Path("/Volumes/PortableSSD").is_dir(),
    }


def _greeks_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    contracts = matrix.get("contracts") or []
    if not contracts:
        return {"n_contracts": 0}
    deltas, gammas, vegas, thetas, rhos, ivs = [], [], [], [], [], []
    call_oi = put_oi = call_vol = put_vol = 0.0
    for c in contracts:
        side = str(c.get("put_call") or c.get("type") or "").lower()
        oi = float(c.get("open_interest") or 0)
        vol = float(c.get("total_volume") or c.get("volume") or 0)
        if "c" in side:
            call_oi += oi
            call_vol += vol
        else:
            put_oi += oi
            put_vol += vol
        for arr, key in (
            (deltas, "delta"),
            (gammas, "gamma"),
            (vegas, "vega"),
            (thetas, "theta"),
            (rhos, "rho"),
            (ivs, "volatility"),
        ):
            v = c.get(key)
            if v is not None:
                try:
                    arr.append(float(v))
                except (TypeError, ValueError):
                    pass

    def _stats(xs: list[float]) -> dict[str, float | None]:
        if not xs:
            return {"mean": None, "sum": None}
        a = np.asarray(xs, dtype=float)
        return {"mean": float(np.nanmean(a)), "sum": float(np.nansum(a))}

    return {
        "n_contracts": len(contracts),
        "call_oi": call_oi,
        "put_oi": put_oi,
        "call_volume": call_vol,
        "put_volume": put_vol,
        "delta": _stats(deltas),
        "gamma": _stats(gammas),
        "vega": _stats(vegas),
        "theta": _stats(thetas),
        "rho": _stats(rhos),
        "iv": _stats(ivs),
        "underlying_price": matrix.get("underlying_price"),
    }


def _data_source_snapshot(symbol: str, as_of: str) -> dict[str, Any]:
    """Denote data-source adapters used for this scan (catalog from DATA_SOURCES)."""
    out: dict[str, Any] = {
        "options.alpaca": {"used": True, "as_of": as_of},
        "market.ohlcv": {"used": False},
        "macro.fred": {"used": False},
        "structure.vol_indices": {"used": False},
        "calendar.opex": {"used": False},
    }
    try:
        from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv

        ohlcv = fetch_daily_ohlcv(symbol)
        out["market.ohlcv"] = {
            "used": True,
            "n_bars": int(len(ohlcv)),
            "last": str(ohlcv.index[-1]) if len(ohlcv) else None,
        }
    except Exception as exc:  # noqa: BLE001
        out["market.ohlcv"] = {"used": False, "error": str(exc)}
    try:
        from gamma_squeeze.features.macro_features import compute_macro_features

        macro = compute_macro_features(as_of=as_of)
        md = macro.to_dict() if hasattr(macro, "to_dict") else dict(macro)
        out["macro.fred"] = {"used": True, "keys": list(md.keys())[:12]}
        out["structure.vol_indices"] = {
            "used": True,
            "vix": md.get("vix"),
            "vvix": md.get("vvix"),
        }
    except Exception as exc:  # noqa: BLE001
        out["macro.fred"] = {"used": False, "error": str(exc)}
    try:
        from gamma_squeeze.data_sources.calendar.opex import OPEXAdapter

        op = OPEXAdapter().fetch(as_of=as_of)
        out["calendar.opex"] = {"used": True, "data": op.get("data") if isinstance(op, dict) else op}
    except Exception as exc:  # noqa: BLE001
        out["calendar.opex"] = {"used": False, "error": str(exc)}
    return out


def _confirmation(
    *,
    gex_row: dict[str, Any],
    patterns: dict[str, Any],
    greeks: dict[str, Any],
    composite: dict[str, Any] | None,
) -> tuple[bool, float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    net_gex = float(gex_row.get("net_gex") or 0.0)
    regime = str(gex_row.get("regime") or "")
    flip_dist = gex_row.get("flip_dist_pct")
    call_wall_dist = gex_row.get("call_wall_dist_pct")

    if net_gex < 0:
        score += 0.22
        reasons.append("negative_net_gex")
    if regime in ("negative_gamma", "Negative Gamma") or "neg" in regime.lower():
        score += 0.12
        reasons.append("negative_gamma_regime")
    if flip_dist is not None and abs(float(flip_dist)) < 0.03:
        score += 0.10
        reasons.append("near_gamma_flip")
    if call_wall_dist is not None and 0 <= float(call_wall_dist) < 0.05:
        score += 0.08
        reasons.append("approaching_call_wall")

    pats = patterns.get("patterns") or []
    breakouts = [
        p
        for p in pats
        if p.get("pattern") in BREAKOUT_PATTERNS
        or "breakout" in str(p.get("pattern") or "").lower()
        or (p.get("target_projection") or {}).get("direction") == "up"
        and float(p.get("confidence") or 0) >= 0.55
    ]
    candles = [p for p in pats if p.get("pattern") in {
        "Hammer", "Bullish Engulfing", "Morning Star", "Three White Soldiers", "Piercing Line"
    }]
    if breakouts:
        score += 0.14
        reasons.append(f"breakout:{breakouts[0].get('pattern')}")
    if candles:
        score += 0.08
        reasons.append(f"candle:{candles[0].get('pattern')}")
    if patterns.get("summary", {}).get("compression"):
        score += 0.06
        reasons.append("volatility_squeeze_pattern")

    gamma_sum = (greeks.get("gamma") or {}).get("sum")
    if gamma_sum is not None and abs(float(gamma_sum)) > 0:
        # large absolute dealer gamma exposure
        score += 0.05
        reasons.append("elevated_gamma_exposure")

    if composite:
        prob = float(composite.get("probability") or 0.0)
        if prob >= 0.55:
            score += 0.18
            reasons.append(f"composite_p={prob:.2f}")
        regime_name = str((composite.get("inputs") or {}).get("regime") or composite.get("regime") or "")
        if regime_name in SQUEEZE_REGIMES:
            score += 0.08
            reasons.append(f"squeeze_regime:{regime_name}")
        gs = float((composite.get("scores") or {}).get("gamma_squeeze_score") or 0.0)
        if gs >= 0.55:
            score += 0.10
            reasons.append(f"gamma_squeeze_score={gs:.2f}")

    score = float(min(1.0, score))
    return score >= CONFIRMED_THRESHOLD, score, reasons


def scan_symbol_confirmed_squeeze(
    symbol: str,
    *,
    as_of: str | None = None,
    market_cap: float | None = None,
    matrix_root: Path | None = None,
    run_composite: bool = True,
    include_data_sources: bool = True,
) -> ConfirmedSqueezeHit:
    sym = symbol.upper()
    root = matrix_root or resolve_matrix_root()
    dates = list_local_dates(root, sym)
    date_used = as_of or (dates[-1] if dates else None)
    if not date_used:
        # try sibling alpaca store directly
        alt = DEFAULT_ALPACA_MATRIX_STORE / sym
        if alt.is_dir():
            dates = sorted(p.stem for p in alt.glob("*.json"))
            date_used = dates[-1] if dates else None
            if date_used:
                matrix = json.loads((alt / f"{date_used}.json").read_text())
            else:
                return ConfirmedSqueezeHit(
                    symbol=sym, as_of="", confirmed=False, confirmation_score=0.0,
                    market_cap=market_cap, error="no_matrix",
                )
        else:
            return ConfirmedSqueezeHit(
                symbol=sym, as_of="", confirmed=False, confirmation_score=0.0,
                market_cap=market_cap, error="no_matrix",
            )
    else:
        matrix = load_local_matrix(root, sym, date_used)
        if not matrix:
            alt = DEFAULT_ALPACA_MATRIX_STORE / sym / f"{date_used}.json"
            if alt.is_file():
                matrix = json.loads(alt.read_text())
            else:
                return ConfirmedSqueezeHit(
                    symbol=sym, as_of=date_used, confirmed=False, confirmation_score=0.0,
                    market_cap=market_cap, error="matrix_missing",
                )

    try:
        gex_row = gex_feature_row(matrix, as_of_date=str(date_used))
    except Exception as exc:  # noqa: BLE001
        return ConfirmedSqueezeHit(
            symbol=sym, as_of=str(date_used), confirmed=False, confirmation_score=0.0,
            market_cap=market_cap, error=f"gex:{exc}",
        )

    greeks = _greeks_summary(matrix)
    dex = {
        "net_dex": gex_row.get("net_dex"),
        "call_dex": gex_row.get("call_dex"),
        "put_dex": gex_row.get("put_dex"),
    }
    try:
        om = compute_options_metrics(matrix, as_of=str(date_used))
        options_metrics = {
            k: v
            for k, v in om.to_dict().items()
            if k
            not in (
                "risk_neutral_density",
                "smile_curve",
                "term_curve",
            )
        }
    except Exception as exc:  # noqa: BLE001
        options_metrics = {"error": str(exc)}

    patterns: dict[str, Any] = {"patterns": [], "summary": {}}
    try:
        from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv

        ohlcv = fetch_daily_ohlcv(sym)
        patterns = detect_patterns(ohlcv, include_candles=True)
    except Exception as exc:  # noqa: BLE001
        patterns = {"patterns": [], "summary": {}, "error": str(exc)}

    pats = patterns.get("patterns") or []
    breakouts = [
        p
        for p in pats
        if p.get("pattern") in BREAKOUT_PATTERNS
        or "breakout" in str(p.get("pattern") or "").lower()
        or str(p.get("pattern")) in ("Gap", "Channel", "Ascending Triangle", "Bull Flag")
    ]
    from gamma_squeeze.patterns.recognition import CANDLESTICK_PATTERNS

    candlesticks = [p for p in pats if p.get("pattern") in CANDLESTICK_PATTERNS]

    composite: dict[str, Any] = {}
    if run_composite:
        try:
            from gamma_squeeze.features.feature_store import build_features_for_symbol, load_features
            from gamma_squeeze.models.composite_squeeze import run_composite_squeeze

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
            if not feat.empty:
                result = run_composite_squeeze(sym, feat, matrix)
                composite = {
                    "probability": result.probability,
                    "magnitude": result.magnitude,
                    "confidence": result.confidence,
                    "risk_rating": result.risk_rating,
                    "scores": result.scores.to_dict(),
                    "inputs": {
                        k: result.inputs.get(k)
                        for k in ("regime", "pattern_breakout", "sentiment")
                        if k in result.inputs
                    },
                }
        except Exception as exc:  # noqa: BLE001
            composite = {"error": str(exc)}

    confirmed, score, reasons = _confirmation(
        gex_row=gex_row,
        patterns=patterns,
        greeks=greeks,
        composite=composite if composite and "error" not in composite else None,
    )
    sources = _data_source_snapshot(sym, str(date_used)) if include_data_sources else {}

    return ConfirmedSqueezeHit(
        symbol=sym,
        as_of=str(date_used),
        confirmed=confirmed,
        confirmation_score=score,
        reasons=reasons,
        market_cap=market_cap,
        gex={
            "net_gex": gex_row.get("net_gex"),
            "call_gex": gex_row.get("call_gex"),
            "put_gex": gex_row.get("put_gex"),
            "gamma_flip": gex_row.get("gamma_flip"),
            "call_wall": gex_row.get("call_wall"),
            "put_wall": gex_row.get("put_wall"),
            "regime": gex_row.get("regime"),
            "flip_dist_pct": gex_row.get("flip_dist_pct"),
            "call_wall_dist_pct": gex_row.get("call_wall_dist_pct"),
            "put_wall_dist_pct": gex_row.get("put_wall_dist_pct"),
            "spot": gex_row.get("spot"),
            "pcr_oi": gex_row.get("pcr_oi"),
        },
        dex=dex,
        greeks=greeks,
        options_metrics=options_metrics,
        patterns={
            "summary": patterns.get("summary") or {},
            "top": pats[:8],
        },
        breakouts=breakouts[:6],
        candlesticks=candlesticks[:6],
        data_sources=sources,
        composite=composite,
    )


def scan_top_market_cap(
    *,
    top: int = 100,
    refresh_universe: bool = False,
    sync_matrices: bool = True,
    run_composite: bool = True,
    confirmed_only: bool = False,
    limit: int | None = None,
    export_root: Path | None = None,
) -> dict[str, Any]:
    """Full universe scan + export under Gamma Squeeze Matrix on portable SSD / fallback."""
    seed_everything()
    universe = load_top_market_cap(top=top, refresh=refresh_universe)
    if limit:
        universe = universe[:limit]
    symbols = [str(u["symbol"]).upper() for u in universe]
    cap_map = {str(u["symbol"]).upper(): float(u.get("market_cap") or 0) for u in universe}

    matrix_root = resolve_matrix_root()
    sync_info: dict[str, Any] = {}
    if sync_matrices:
        sync_info = sync_matrices_to_ssd(symbols, dest_root=matrix_root, latest_only=True)

    hits: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []
    # Pass 1: GEX/DEX/greeks/patterns (fast). Pass 2: composite on near-confirmed.
    for i, sym in enumerate(symbols, 1):
        log_event(logger, "scan_symbol", symbol=sym, index=i, total=len(symbols), pass_=1)
        hit = scan_symbol_confirmed_squeeze(
            sym,
            market_cap=cap_map.get(sym),
            matrix_root=matrix_root,
            run_composite=False,
            include_data_sources=True,
        )
        d = hit.to_dict()
        if run_composite and (hit.confirmed or hit.confirmation_score >= 0.40) and not hit.error:
            log_event(logger, "scan_composite", symbol=sym, score=hit.confirmation_score)
            hit2 = scan_symbol_confirmed_squeeze(
                sym,
                as_of=hit.as_of,
                market_cap=cap_map.get(sym),
                matrix_root=matrix_root,
                run_composite=True,
                include_data_sources=False,
            )
            # merge richer confirmation + composite; keep pass-1 data_sources/patterns
            d2 = hit2.to_dict()
            d["composite"] = d2.get("composite") or {}
            d["confirmed"] = d2.get("confirmed")
            d["confirmation_score"] = d2.get("confirmation_score")
            d["reasons"] = d2.get("reasons") or d.get("reasons")
        hits.append(d)
        if d.get("confirmed"):
            confirmed.append(d)

    export_base = export_root or (matrix_root / "scans" / "confirmed_gamma_squeeze")
    export_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": "top_nasdaq_by_market_cap",
        "top_n": top,
        "n_scanned": len(hits),
        "n_confirmed": len(confirmed),
        "confirmed_threshold": CONFIRMED_THRESHOLD,
        "matrix_root": str(matrix_root),
        "ssd_mounted": Path("/Volumes/PortableSSD").is_dir(),
        "sync": sync_info,
        "variables": [
            "candlestick_patterns",
            "breakouts",
            "options_matrix_greeks",
            "gex",
            "dex",
            "options_metrics",
            "data_sources",
            "composite_squeeze",
        ],
        "confirmed": confirmed if confirmed_only else confirmed,
        "results": confirmed if confirmed_only else hits,
        "universe_tickers": universe,
    }
    out_path = export_base / f"scan_{stamp}.json"
    latest = export_base / "latest.json"
    confirmed_path = export_base / f"confirmed_{stamp}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    with latest.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    with confirmed_path.open("w") as f:
        json.dump(
            {
                "generated_at": payload["generated_at"],
                "n_confirmed": len(confirmed),
                "confirmed": confirmed,
            },
            f,
            indent=2,
            default=str,
        )
    # index
    with (export_base / "index.json").open("w") as f:
        json.dump(
            {
                "latest": str(latest),
                "last_scan": str(out_path),
                "last_confirmed": str(confirmed_path),
                "n_confirmed": len(confirmed),
                "updated_at": payload["generated_at"],
            },
            f,
            indent=2,
        )

    return {
        "export_dir": str(export_base),
        "scan_path": str(out_path),
        "confirmed_path": str(confirmed_path),
        "latest_path": str(latest),
        "n_scanned": len(hits),
        "n_confirmed": len(confirmed),
        "confirmed_symbols": [c["symbol"] for c in confirmed],
        "ssd_mounted": payload["ssd_mounted"],
        "matrix_root": str(matrix_root),
        "sync": sync_info,
    }
