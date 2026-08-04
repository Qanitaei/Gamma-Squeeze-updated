"""Feature engineering: dealer positioning, options metrics, technicals, macro."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.config import resolve_matrix_root
from gamma_squeeze.features.dealer_positioning import compute_dealer_positioning
from gamma_squeeze.features.feature_store import (
    DEALER_FEATURE_COLUMNS,
    build_features_for_symbol,
    load_features,
)
from gamma_squeeze.features.macro_features import (
    MACRO_FEATURE_COLUMNS,
    MACRO_FEATURES_FIELDS,
    compute_macro_features,
)
from gamma_squeeze.features.options_metrics import (
    OPTIONS_METRIC_COLUMNS,
    OPTIONS_METRICS_FIELDS,
    compute_options_metrics,
)
from gamma_squeeze.features.technical_indicators import (
    TECHNICAL_INDICATOR_COLUMNS,
    TECHNICAL_INDICATORS_FIELDS,
    compute_technical_indicators,
)
from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv


def build_or_load_features(
    symbol: str,
    *,
    rebuild: bool = False,
    include_skew: bool = False,
    include_macro: bool = True,
    include_dealer: bool = True,
    include_options_metrics: bool = True,
    include_technicals: bool = True,
) -> dict[str, Any]:
    sym = symbol.upper()
    if rebuild:
        df = build_features_for_symbol(
            sym,
            include_skew=include_skew,
            include_macro=include_macro,
            include_ohlcv=True,
            include_dealer=include_dealer,
            include_options_metrics=include_options_metrics,
            include_technicals=include_technicals,
        )
    else:
        df = load_features(sym)
        if df.empty:
            df = build_features_for_symbol(
                sym,
                include_skew=include_skew,
                include_macro=include_macro,
                include_ohlcv=True,
                include_dealer=include_dealer,
                include_options_metrics=include_options_metrics,
                include_technicals=include_technicals,
            )
    latest = df.sort_values("as_of").iloc[-1].to_dict() if not df.empty else {}
    dealer_cols = [c for c in DEALER_FEATURE_COLUMNS if c in df.columns]
    options_cols = [c for c in OPTIONS_METRIC_COLUMNS if c in df.columns]
    tech_cols = [c for c in TECHNICAL_INDICATOR_COLUMNS if c in df.columns]
    macro_cols = [c for c in MACRO_FEATURE_COLUMNS if c in df.columns]
    return {
        "symbol": sym,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "latest": {k: (None if (isinstance(v, float) and v != v) else v) for k, v in latest.items()},
        "as_of": latest.get("as_of"),
        "groups": {
            "dealer_positioning": dealer_cols,
            "options_metrics": options_cols,
            "technical_indicators": tech_cols + [c for c in ("ret_1d", "ret_5d", "rvol_10d") if c in df.columns],
            "macro_features": macro_cols + [c for c in ("fedfunds", "dgs10", "t10y2y") if c in df.columns],
            "vol_surface": [
                c for c in df.columns if c in ("iv_atm", "skew_25d", "term_slope", "iv_minus_hv", "call_bias")
            ],
        },
    }


def _load_options_matrix(symbol: str, as_of: str | None = None) -> tuple[dict[str, Any] | None, str | None, str]:
    """Load options matrix: local SSD → Alpaca API → alpaca-options-matrix-backup KV."""
    sym = symbol.upper()
    root = resolve_matrix_root()
    dates = list_local_dates(root, sym)
    date_used = as_of or (dates[-1] if dates else None)
    if date_used:
        matrix = load_local_matrix(root, sym, date_used)
        if matrix:
            return matrix, date_used, "ssd"

    # Alpaca adapter: live API then KV namespace alpaca-options-matrix-backup
    from gamma_squeeze.data_sources.registry import get_adapter, load_all_adapters

    load_all_adapters()
    out = get_adapter("options.alpaca").fetch_matrix(sym, as_of=as_of)
    matrix = (out.get("data") or {}).get("matrix")
    if matrix:
        origin = (out.get("meta") or {}).get("origin", "alpaca")
        used = out.get("as_of") or as_of or matrix.get("as_of_date")
        return matrix, used, str(origin)
    return None, as_of, "none"


def compute_dealer_features(
    symbol: str,
    *,
    as_of: str | None = None,
    include_gamma_by_strike: bool = True,
) -> dict[str, Any]:
    sym = symbol.upper()
    root = resolve_matrix_root()
    dates = list_local_dates(root, sym)
    matrix, date_used, origin = _load_options_matrix(sym, as_of=as_of)
    if not matrix or not date_used:
        return {
            "symbol": sym,
            "error": "no_options_matrix",
            "success": False,
            "hint": "Configure ALPACA_API_KEY_ID/SECRET or sync alpaca-options-matrix-backup",
        }
    prior = None
    if date_used in dates:
        i = dates.index(date_used)
        if i > 0:
            prior_matrix = load_local_matrix(root, sym, dates[i - 1])
            if prior_matrix:
                prior = compute_dealer_positioning(prior_matrix, as_of=dates[i - 1])
    feats = compute_dealer_positioning(matrix, as_of=date_used, prior=prior)
    payload = feats.to_dict()
    if not include_gamma_by_strike:
        payload.pop("gamma_by_strike", None)
    return {
        "symbol": sym,
        "as_of": date_used,
        "success": True,
        "matrix_origin": origin,
        "dealer_positioning": payload,
        "feature_names": DEALER_FEATURE_COLUMNS,
    }


def compute_options_metrics_features(
    symbol: str,
    *,
    as_of: str | None = None,
    include_curves: bool = True,
) -> dict[str, Any]:
    """Options Metrics from Alpaca API → SSD → alpaca-options-matrix-backup KV."""
    sym = symbol.upper()
    root = resolve_matrix_root()
    dates = list_local_dates(root, sym)
    matrix, date_used, origin = _load_options_matrix(sym, as_of=as_of)
    if not matrix or not date_used:
        return {
            "symbol": sym,
            "error": "no_options_matrix",
            "success": False,
            "hint": "Configure ALPACA_API_KEY_ID/SECRET or sync alpaca-options-matrix-backup",
        }

    prior_oi = None
    atm_hist: list[float] = []
    if date_used in dates:
        i = dates.index(date_used)
        for prev in dates[max(0, i - 40) : i]:
            pm = load_local_matrix(root, sym, prev)
            if not pm:
                continue
            try:
                tmp = compute_options_metrics(pm, as_of=prev)
                prior_oi = tmp.open_interest
                if tmp.atm_iv is not None:
                    atm_hist.append(float(tmp.atm_iv))
            except Exception:  # noqa: BLE001
                continue

    try:
        ohlcv = fetch_daily_ohlcv(sym)
    except Exception:  # noqa: BLE001
        ohlcv = None

    feats = compute_options_metrics(
        matrix,
        as_of=date_used,
        ohlcv=ohlcv,
        prior_oi=prior_oi,
        atm_iv_history=atm_hist,
    )
    payload = feats.to_dict()
    if not include_curves:
        payload.pop("smile_curve", None)
        payload.pop("term_curve", None)
        # Keep risk_neutral_density — it is a locked Options Metrics field
    return {
        "symbol": sym,
        "as_of": date_used,
        "success": True,
        "matrix_origin": origin,
        "options_metrics": payload,
        "feature_names": list(OPTIONS_METRICS_FIELDS),
        "ml_columns": OPTIONS_METRIC_COLUMNS,
    }


def compute_technical_features(
    symbol: str,
    *,
    as_of: str | None = None,
    anchor_date: str | None = None,
    include_volume_profile: bool = True,
) -> dict[str, Any]:
    """Technical Indicators from OHLCV (Alpaca / OHLCV worker)."""
    sym = symbol.upper()
    try:
        ohlcv = fetch_daily_ohlcv(sym)
    except Exception as exc:  # noqa: BLE001
        return {"symbol": sym, "success": False, "error": str(exc)}
    if ohlcv.empty:
        return {"symbol": sym, "success": False, "error": "empty_ohlcv"}
    date_used = as_of
    if date_used is None:
        date_used = str(ohlcv.index[-1])[:10]
    feats = compute_technical_indicators(
        ohlcv,
        symbol=sym,
        as_of=date_used,
        anchor_date=anchor_date,
    )
    payload = feats.catalog_dict()
    if not include_volume_profile:
        vp = payload.get("volume_profile")
        if isinstance(vp, dict):
            vp.pop("bins", None)
    return {
        "symbol": sym,
        "as_of": date_used,
        "success": True,
        "ohlcv_bars": int(len(ohlcv)),
        "technical_indicators": payload,
        "feature_names": list(TECHNICAL_INDICATORS_FIELDS),
        "ml_columns": TECHNICAL_INDICATOR_COLUMNS,
    }


def compute_macro_feature_block(
    *,
    as_of: str | None = None,
    include_external_vol: bool = True,
) -> dict[str, Any]:
    """Macro Features from Fred-Economic-data KV (+ CBOE/structure fallbacks)."""
    try:
        feats = compute_macro_features(as_of=as_of, include_external_vol=include_external_vol)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc), "as_of": as_of}
    payload = feats.catalog_dict()
    return {
        "as_of": feats.as_of,
        "success": True,
        "macro_features": payload,
        "feature_names": list(MACRO_FEATURES_FIELDS),
        "ml_columns": MACRO_FEATURE_COLUMNS,
        "namespace": "Fred-Economic-data",
    }
