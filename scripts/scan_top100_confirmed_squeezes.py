#!/usr/bin/env python3
"""Scan top-N NASDAQ names by market cap for confirmed gamma squeezes.

Uses Nasdaq screener market-cap ranking via
`gamma_squeeze.ingest.nasdaq_universe.top_nasdaq_by_market_cap`.

For each ticker denotes:
  - candlestick patterns & breakouts
  - options matrix Greeks (delta/gamma/vega/theta/rho/IV)
  - GEX / DEX
  - options metrics + data-source adapters (DATA_SOURCES catalog)

Exports JSON under portable SSD Gamma Squeeze Matrix (fallback local if unmounted).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gamma_squeeze.scan.confirmed_squeeze import scan_top_market_cap  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", type=int, default=100, help="Top N by market cap (default 100)")
    p.add_argument("--limit", type=int, default=0, help="Optional cap for smoke runs")
    p.add_argument("--refresh-universe", action="store_true", help="Re-rank via Nasdaq screener")
    p.add_argument("--no-sync", action="store_true", help="Skip copying matrices into SSD tree")
    p.add_argument("--no-composite", action="store_true", help="Skip composite engine (faster)")
    p.add_argument("--confirmed-only", action="store_true")
    args = p.parse_args()

    summary = scan_top_market_cap(
        top=args.top,
        refresh_universe=args.refresh_universe,
        sync_matrices=not args.no_sync,
        run_composite=not args.no_composite,
        confirmed_only=args.confirmed_only,
        limit=args.limit or None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
