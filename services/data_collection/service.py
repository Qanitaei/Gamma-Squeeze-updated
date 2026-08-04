"""Data collection domain adapter — Alpaca-forward (no Schwab callbacks)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.config import (
    alpaca_backup_worker_url,
    alpaca_configured,
    alpaca_credentials,
    resolve_matrix_root,
)
from gamma_squeeze.ingest.alpaca_backup_client import (
    health as backup_health,
    list_local_dates,
    list_matrix_dates,
    list_symbols,
    load_local_matrix,
)
from gamma_squeeze.ingest.macro_client import fetch_macro_snapshot
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv


def collect_snapshot(symbol: str, *, as_of: str | None = None, lookback_days: int = 60) -> dict[str, Any]:
    sym = symbol.upper()
    root = resolve_matrix_root()
    local_dates = list_local_dates(root, sym)
    matrix = None
    source = "none"
    date_used = as_of
    errors: list[str] = []

    # 1) Live Alpaca API
    if alpaca_configured():
        try:
            from gamma_squeeze.ingest.alpaca_api import alpaca_api_health, fetch_live_options_matrix

            live = fetch_live_options_matrix(sym, as_of=as_of)
            if live.get("contracts"):
                matrix = live
                source = "alpaca_api"
                date_used = live.get("as_of_date") or as_of
            api_health = alpaca_api_health()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"alpaca_api:{exc}")
            api_health = {"ok": False, "error": str(exc), "configured": True}
    else:
        api_health = {"ok": False, "configured": False}

    # 2) SSD
    if matrix is None:
        if as_of:
            matrix = load_local_matrix(root, sym, as_of)
            if matrix:
                source = "ssd"
                date_used = as_of
        elif local_dates:
            date_used = local_dates[-1]
            matrix = load_local_matrix(root, sym, date_used)
            if matrix:
                source = "ssd"

    kv_dates: list[str] = []
    try:
        kv_dates = list_matrix_dates(sym)[-lookback_days:]
    except Exception as exc:  # noqa: BLE001
        kv_dates_error = str(exc)
    else:
        kv_dates_error = ""

    ohlcv_meta: dict[str, Any] = {}
    try:
        ohlcv = fetch_daily_ohlcv(sym)
        ohlcv_meta = {
            "n_bars": int(len(ohlcv)),
            "last_date": str(ohlcv.index[-1]) if len(ohlcv) else None,
            "origin": "alpaca_or_kv",
        }
    except Exception as exc:  # noqa: BLE001
        ohlcv_meta = {"error": str(exc)}

    try:
        macro = fetch_macro_snapshot()
    except Exception as exc:  # noqa: BLE001
        macro = {"error": str(exc)}

    creds = alpaca_credentials()
    return {
        "symbol": sym,
        "as_of": date_used,
        "matrix_source": source,
        "matrix_present": matrix is not None,
        "matrix_contracts": len((matrix or {}).get("contracts") or []) if matrix else 0,
        "local_dates": local_dates[-lookback_days:],
        "kv_dates": kv_dates,
        "kv_dates_error": kv_dates_error,
        "ssd_root": str(root),
        "backup_worker": alpaca_backup_worker_url(),
        "backup_health": backup_health(),
        "alpaca": {
            "configured": alpaca_configured(),
            "trading_base": creds.get("trading_base"),
            "data_base": creds.get("data_base"),
            "paper": creds.get("paper"),
            "health": api_health,
        },
        "ohlcv": ohlcv_meta,
        "macro": macro,
        "errors": errors,
        "available_symbols_sample": (list_symbols() or ([sym] if local_dates else []))[:20],
    }
