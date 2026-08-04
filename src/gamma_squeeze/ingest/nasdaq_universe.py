"""NASDAQ ticker universe and market-cap ranking."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

NASDAQ_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=25&offset=0&download=true&exchange=nasdaq"
)
NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
}


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_nasdaq_tickers() -> list[dict[str, str | float]]:
    req = urllib.request.Request(NASDAQ_SCREENER_URL, headers=NASDAQ_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download NASDAQ ticker list: {exc}") from exc

    rows = (payload.get("data") or {}).get("rows") or []
    tickers: list[dict[str, str | float]] = []
    seen: set[str] = set()
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cap_raw = str(row.get("marketCap", "0")).replace(",", "").strip()
        try:
            market_cap = float(cap_raw)
        except ValueError:
            market_cap = 0.0
        tickers.append(
            {
                "symbol": symbol,
                "name": str(row.get("name", "")).strip(),
                "sector": str(row.get("sector", "")).strip(),
                "industry": str(row.get("industry", "")).strip(),
                "market_cap": market_cap,
            }
        )
    tickers.sort(key=lambda item: str(item["symbol"]))
    return tickers


def top_nasdaq_by_market_cap(
    client: Any | None = None,
    n: int = 100,
    *,
    throttle_s: float = 0.55,
    cache_path: Path | None = None,
) -> list[dict[str, str]]:
    """Return top *n* NASDAQ symbols using Nasdaq screener marketCap (fast)."""
    del client, throttle_s  # retained for call-site compatibility
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(cached) >= n and float(cached[0].get("market_cap") or 0) > 0:
            return cached[:n]

    universe = fetch_nasdaq_tickers()
    with_cap = [t for t in universe if float(t.get("market_cap") or 0) > 0]
    ranked = sorted(with_cap, key=lambda t: float(t["market_cap"]), reverse=True)
    top = ranked[:n]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(top, indent=2), encoding="utf-8")
    return top
