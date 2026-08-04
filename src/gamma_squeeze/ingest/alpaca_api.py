"""Direct Alpaca REST pulls for options matrix / OHLCV."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from gamma_squeeze.config import alpaca_configured, alpaca_credentials
from gamma_squeeze.ingest.alpaca_client import AlpacaClient
from gamma_squeeze.ingest.alpaca_options_matrix import (
    _group_rows_by_expiration,
    build_options_rows,
)


def _client() -> AlpacaClient:
    """Return AlpacaClient when credentials are present."""
    if not alpaca_configured():
        raise RuntimeError("Alpaca API keys not configured (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)")
    creds = alpaca_credentials()
    return AlpacaClient(
        key_id=creds["key_id"],
        secret=creds["secret"],
        data_base=creds["data_base"],
        trading_base=creds["trading_base"],
        option_feed=creds.get("option_feed") or "indicative",
    )


def fetch_live_options_matrix(
    symbol: str,
    *,
    as_of: str | None = None,
    max_dte_days: int = 120,
) -> dict[str, Any]:
    """Pull options chain/matrix from Alpaca Market Data API."""
    client = _client()
    sym = symbol.upper()
    as_of_date = None
    if as_of:
        as_of_date = datetime.strptime(as_of[:10], "%Y-%m-%d").date()
    rows, meta = build_options_rows(
        client,
        sym,
        as_of_date=as_of_date,
        max_dte_days=max_dte_days,
        historical_require_activity=bool(as_of_date),
    )
    by_exp = _group_rows_by_expiration(rows) if rows else []
    as_of_out = (as_of_date or datetime.now(timezone.utc).date()).isoformat()
    return {
        "symbol": sym,
        "source": "alpaca_api",
        "as_of_date": as_of_out,
        "underlying_price": meta.get("underlying_price"),
        "contracts": rows,
        "by_expiration": by_exp,
        "meta": {**meta, "origin": "alpaca_api"},
    }


def fetch_stock_ohlcv(
    symbol: str,
    *,
    lookback_days: int = 252,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    """Daily (or other) OHLCV bars from Alpaca data API."""
    client = _client()
    sym = symbol.upper()
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(lookback_days * 1.6) + 5)
    payload = client.get_data(
        "/v2/stocks/bars",
        {
            "symbols": sym,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": "split",
            "feed": "iex",
        },
    )
    bars = (payload.get("bars") or {}).get(sym) or []
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows = []
    for b in bars:
        ts = str(b.get("t") or "")[:10]
        rows.append(
            {
                "date": ts,
                "open": float(b.get("o") or 0),
                "high": float(b.get("h") or 0),
                "low": float(b.get("l") or 0),
                "close": float(b.get("c") or 0),
                "volume": float(b.get("v") or 0),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    df = df.set_index("date")
    return df.tail(lookback_days)


def fetch_latest_trade(symbol: str) -> dict[str, Any]:
    client = _client()
    payload = client.get_data(f"/v2/stocks/{symbol.upper()}/trades/latest")
    return payload.get("trade") or payload


def fetch_latest_quote(symbol: str) -> dict[str, Any]:
    client = _client()
    payload = client.get_data(f"/v2/stocks/{symbol.upper()}/quotes/latest")
    return payload.get("quote") or payload


def alpaca_api_health() -> dict[str, Any]:
    creds = alpaca_credentials()
    out: dict[str, Any] = {
        "configured": alpaca_configured(),
        "paper": creds.get("paper"),
        "trading_base": creds.get("trading_base"),
        "data_base": creds.get("data_base"),
    }
    if not alpaca_configured():
        out["ok"] = False
        return out
    try:
        client = _client()
        acct = client.verify()
        out["ok"] = True
        out["account_status"] = acct.get("status") or acct.get("account_number") or "verified"
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = str(exc)
    return out
