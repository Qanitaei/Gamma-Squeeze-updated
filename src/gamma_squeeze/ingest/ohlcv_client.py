"""OHLCV fetch — Alpaca API first, then Cloudflare KV worker fallback."""

from __future__ import annotations

import json
import ssl
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from gamma_squeeze.config import alpaca_configured, ohlcv_worker_url

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def _fetch_json(url: str, *, timeout: int = 120) -> Any:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "GammaSqueezePlatform/0.1"})
    with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_daily_ohlcv(symbol: str, *, base_url: str | None = None) -> pd.DataFrame:
    """Return daily OHLCV DataFrame indexed by date string YYYY-MM-DD."""
    sym = symbol.upper()
    errors: list[str] = []

    # 1) Alpaca Market Data API
    if alpaca_configured():
        try:
            from gamma_squeeze.ingest.alpaca_api import fetch_stock_ohlcv

            df = fetch_stock_ohlcv(sym, lookback_days=252, timeframe="1Day")
            if not df.empty:
                return df
            errors.append("alpaca_empty")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"alpaca:{exc}")

    # 2) KV / worker fallback
    base = (base_url or ohlcv_worker_url()).rstrip("/")
    candidates = [
        f"{base}/v1/{sym}/ohlcv/1d_rth",
        f"{base}/{sym}/ohlcv/1d_rth",
        f"{base}/v1/{sym}/1d",
        f"{base}/{sym}",
    ]
    payload: Any = None
    last_err: Exception | None = None
    for url in candidates:
        try:
            payload = _fetch_json(url)
            break
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            continue
    if payload is None:
        raise RuntimeError(f"OHLCV fetch failed for {sym}: {last_err}; prior={errors}")

    rows = _extract_bars(payload)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    df["date"] = df["date"].astype(str).str[:10]
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _extract_bars(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_normalize_bar(x) for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("bars", "data", "ohlcv", "records", "rows"):
        nested = payload.get(key)
        if isinstance(nested, list):
            return [_normalize_bar(x) for x in nested if isinstance(x, dict)]
        if isinstance(nested, dict) and isinstance(nested.get("bars"), list):
            return [_normalize_bar(x) for x in nested["bars"] if isinstance(x, dict)]
    for key, val in payload.items():
        if "1d" in str(key).lower() and isinstance(val, list):
            return [_normalize_bar(x) for x in val if isinstance(x, dict)]
    return []


def _normalize_bar(raw: dict[str, Any]) -> dict[str, Any]:
    date = (
        raw.get("date")
        or raw.get("datetime")
        or raw.get("timestamp")
        or raw.get("t")
        or raw.get("time")
    )
    return {
        "date": str(date)[:10] if date else "",
        "open": raw.get("open", raw.get("o")),
        "high": raw.get("high", raw.get("h")),
        "low": raw.get("low", raw.get("l")),
        "close": raw.get("close", raw.get("c")),
        "volume": raw.get("volume", raw.get("v")),
    }
