"""Dated feature tables persisted under Gamma Squeeze Matrix / features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gamma_squeeze.config import resolve_features_root, resolve_matrix_root
from gamma_squeeze.features.dealer_positioning import (
    DealerPositioningFeatures,
    compute_dealer_positioning,
)
from gamma_squeeze.features.gex_features import gex_feature_row
from gamma_squeeze.features.options_metrics import (
    OPTIONS_METRIC_COLUMNS,
    compute_options_metrics,
)
from gamma_squeeze.features.positioning import enrich_with_positioning
from gamma_squeeze.features.macro_features import (
    MACRO_FEATURE_COLUMNS,
    compute_macro_features,
)
from gamma_squeeze.features.technical_indicators import (
    TECHNICAL_INDICATOR_COLUMNS,
    compute_technical_indicators,
)
from gamma_squeeze.features.vol_surface import fetch_skew_features
from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv


DEALER_FEATURE_COLUMNS = [
    "net_gex",
    "gamma_exposure",
    "gamma_flip",
    "zero_gamma",
    "call_wall",
    "put_wall",
    "dealer_hedge_requirement",
    "dealer_share_imbalance",
    "dealer_delta",
    "dealer_gamma",
    "dealer_charm",
    "dealer_vanna",
    "dealer_vomma",
    "dealer_vega",
    "dealer_theta",
    "dealer_speed",
    "dealer_color",
    "dealer_zomma",
    "dealer_ultima",
    "dealer_cross_gamma",
    "dealer_volga",
    "dealer_elasticity",
    "dealer_convexity",
    "dealer_liquidity_score",
    "dealer_inventory_estimate",
    "dealer_gamma_acceleration",
    "dealer_hedging_velocity",
]

_BASE_FEATURE_COLUMNS = [
    "symbol",
    "as_of",
    "spot",
    "call_gex",
    "put_gex",
    "net_dex",
    "pcr_oi",
    "pcr_vol",
    "regime_neg_gamma",
    "flip_dist_pct",
    "call_wall_dist_pct",
    "put_wall_dist_pct",
    "gex_per_spot",
    "flag_fragile_gamma",
    "flag_near_flip",
    "flag_call_crowding",
    "flag_wall_magnet",
    "positioning_stress",
    "ret_1d",
    "ret_5d",
    "rvol_10d",
    "iv_atm",
    "skew_25d",
    "term_slope",
    "iv_minus_hv",
    "call_bias",
    # legacy macro aliases (kept for older models)
    "fedfunds",
    "dgs10",
    "t10y2y",
]

# Preserve order: core → dealer → options → technicals → macro → rest
FEATURE_COLUMNS = list(
    dict.fromkeys(
        _BASE_FEATURE_COLUMNS[:3]
        + DEALER_FEATURE_COLUMNS
        + OPTIONS_METRIC_COLUMNS
        + TECHNICAL_INDICATOR_COLUMNS
        + MACRO_FEATURE_COLUMNS
        + _BASE_FEATURE_COLUMNS[3:]
    )
)


def features_path(symbol: str, *, root: Path | None = None) -> Path:
    base = root or resolve_features_root()
    return base / "by_ticker" / symbol.upper() / "features.json"


def load_features(symbol: str, *, root: Path | None = None) -> pd.DataFrame:
    path = features_path(symbol, root=root)
    if not path.is_file():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    with path.open() as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = rows.get("rows") or rows.get("features") or []
    df = pd.DataFrame(rows)
    if "as_of" in df.columns:
        df = df.sort_values("as_of")
    return df


def save_features(symbol: str, df: pd.DataFrame, *, root: Path | None = None) -> Path:
    path = features_path(symbol, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": symbol.upper(),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "rows": df.where(pd.notnull(df), None).to_dict(orient="records"),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def build_features_for_symbol(
    symbol: str,
    *,
    matrix_root: Path | None = None,
    features_root: Path | None = None,
    dates: list[str] | None = None,
    include_skew: bool = True,
    include_macro: bool = True,
    include_ohlcv: bool = True,
    include_dealer: bool = True,
    include_options_metrics: bool = True,
    include_technicals: bool = True,
) -> pd.DataFrame:
    """Build dated feature rows preferring local Gamma Squeeze Matrix SSD."""
    mroot = matrix_root or resolve_matrix_root()
    sym = symbol.upper()
    use_dates = dates or list_local_dates(mroot, sym)
    if not use_dates:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    ohlcv = pd.DataFrame()
    if include_ohlcv:
        try:
            ohlcv = fetch_daily_ohlcv(sym)
        except Exception:  # noqa: BLE001
            ohlcv = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    prior_dealer: DealerPositioningFeatures | None = None
    prior_oi: float | None = None
    atm_iv_history: list[float] = []
    macro_by_date: dict[str, dict[str, Any]] = {}
    for d in use_dates:
        matrix = load_local_matrix(mroot, sym, d)
        if not matrix:
            continue
        gex = enrich_with_positioning(gex_feature_row(matrix, as_of_date=d))
        row: dict[str, Any] = {
            "symbol": sym,
            "as_of": d,
            "spot": gex.get("spot"),
            "net_gex": gex.get("net_gex"),
            "call_gex": gex.get("call_gex"),
            "put_gex": gex.get("put_gex"),
            "net_dex": gex.get("net_dex"),
            "pcr_oi": gex.get("pcr_oi"),
            "pcr_vol": gex.get("pcr_vol"),
            "regime_neg_gamma": gex.get("regime_neg_gamma"),
            "flip_dist_pct": gex.get("flip_dist_pct"),
            "call_wall_dist_pct": gex.get("call_wall_dist_pct"),
            "put_wall_dist_pct": gex.get("put_wall_dist_pct"),
            "gex_per_spot": gex.get("gex_per_spot"),
            "flag_fragile_gamma": gex.get("flag_fragile_gamma"),
            "flag_near_flip": gex.get("flag_near_flip"),
            "flag_call_crowding": gex.get("flag_call_crowding"),
            "flag_wall_magnet": gex.get("flag_wall_magnet"),
            "positioning_stress": gex.get("positioning_stress"),
        }

        if include_macro:
            if d not in macro_by_date:
                try:
                    # Point-in-time FRED only during backfill (skip live CBOE overlays)
                    mfeats = compute_macro_features(as_of=d, include_external_vol=False)
                    flat_m = mfeats.flat_features()
                except Exception:  # noqa: BLE001
                    flat_m = {}
                macro_by_date[d] = flat_m
            flat_m = macro_by_date[d]
            for col in MACRO_FEATURE_COLUMNS:
                if col in flat_m:
                    row[col] = flat_m[col]
            # legacy aliases
            row["fedfunds"] = flat_m.get("fed_funds")
            row["dgs10"] = flat_m.get("treasury_10y")
            row["t10y2y"] = flat_m.get("yield_spread")
        else:
            row["fedfunds"] = None
            row["dgs10"] = None
            row["t10y2y"] = None

        if include_dealer:
            dealer = compute_dealer_positioning(matrix, as_of=d, prior=prior_dealer)
            prior_dealer = dealer
            flat = dealer.flat_features()
            for col in DEALER_FEATURE_COLUMNS:
                if col in flat:
                    row[col] = flat[col]

        if include_options_metrics:
            opt = compute_options_metrics(
                matrix,
                as_of=d,
                ohlcv=ohlcv if not ohlcv.empty else None,
                prior_oi=prior_oi,
                atm_iv_history=atm_iv_history,
            )
            prior_oi = opt.open_interest
            if opt.atm_iv is not None:
                atm_iv_history.append(float(opt.atm_iv))
            flat_opt = opt.flat_features()
            for col in OPTIONS_METRIC_COLUMNS:
                if col in flat_opt:
                    row[col] = flat_opt[col]
            # keep atm_iv alias used elsewhere
            if opt.atm_iv is not None:
                row["iv_atm"] = opt.atm_iv
            if opt.skew is not None:
                row["skew_25d"] = opt.skew

        if not ohlcv.empty and d in ohlcv.index:
            close = float(ohlcv.loc[d, "close"])
            idx = list(ohlcv.index)
            i = idx.index(d)
            row["ret_1d"] = _fwd_ret(ohlcv, i, 1)
            row["ret_5d"] = _fwd_ret(ohlcv, i, 5)
            # realized vol lookback uses past closes
            past = ohlcv["close"].iloc[max(0, i - 10) : i + 1].pct_change().dropna()
            row["rvol_10d"] = float(past.std() * (252**0.5)) if len(past) else None
            row["spot"] = close
            if include_technicals:
                hist = ohlcv.loc[:d]
                tech = compute_technical_indicators(hist, symbol=sym, as_of=d)
                flat_tech = tech.flat_features()
                for col in TECHNICAL_INDICATOR_COLUMNS:
                    if col in flat_tech:
                        row[col] = flat_tech[col]
        else:
            row["ret_1d"] = None
            row["ret_5d"] = None
            row["rvol_10d"] = None

        if include_skew:
            try:
                skew = fetch_skew_features(sym, d)
                row.update(skew)
            except Exception:  # noqa: BLE001
                # Keep options-metrics iv/skew if already populated
                for k in ("iv_atm", "skew_25d", "term_slope", "iv_minus_hv", "call_bias"):
                    row.setdefault(k, None)
        else:
            # Do not clobber iv/skew already set by options metrics
            for k in ("iv_atm", "skew_25d", "term_slope", "iv_minus_hv", "call_bias"):
                row.setdefault(k, None)

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        save_features(sym, df, root=features_root)
    return df


def _fwd_ret(ohlcv: pd.DataFrame, i: int, horizon: int) -> float | None:
    """Backward-looking return ending at i (for contemporaneous features)."""
    if i < horizon:
        return None
    c0 = float(ohlcv["close"].iloc[i - horizon])
    c1 = float(ohlcv["close"].iloc[i])
    if c0 <= 0:
        return None
    return (c1 / c0) - 1.0


def model_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Numeric feature columns for sklearn models."""
    skip = {"symbol", "as_of", "regime"}
    preferred = [c for c in FEATURE_COLUMNS if c not in skip and c in df.columns]
    extras = [
        c
        for c in df.columns
        if c not in skip
        and c not in preferred
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    cols = preferred + extras
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, cols
