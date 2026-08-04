#!/usr/bin/env python3
"""Phase 2: Export annual options matrices from alpaca-options-matrix-backup KV
to portable SSD under Gamma Squeeze Matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.config import (  # noqa: E402
    DEFAULT_LOOKBACK_TRADING_DAYS,
    alpaca_backup_worker_url,
    resolve_matrix_root,
)
from gamma_squeeze.ingest.alpaca_backup_client import (  # noqa: E402
    fetch_options_matrix,
    health,
    list_matrix_dates,
    list_symbols,
)


def _filter_dates(
    dates: list[str],
    *,
    lookback_days: int,
    date_from: str | None,
    date_to: str | None,
) -> list[str]:
    out = list(dates)
    if date_from:
        out = [d for d in out if d >= date_from]
    if date_to:
        out = [d for d in out if d <= date_to]
    if lookback_days > 0 and len(out) > lookback_days:
        out = out[-lookback_days:]
    return out


def export_symbol(
    symbol: str,
    *,
    root: Path,
    base_url: str,
    lookback_days: int,
    date_from: str | None,
    date_to: str | None,
    resume: bool,
    throttle_s: float,
) -> dict:
    sym = symbol.upper()
    ticker_dir = root / "tickers" / sym
    ticker_dir.mkdir(parents=True, exist_ok=True)
    dates = list_matrix_dates(sym, base_url=base_url)
    dates = _filter_dates(dates, lookback_days=lookback_days, date_from=date_from, date_to=date_to)

    written = 0
    skipped = 0
    errors: list[str] = []
    for d in dates:
        out_path = ticker_dir / f"{d}.json"
        if resume and out_path.is_file() and out_path.stat().st_size > 50:
            skipped += 1
            continue
        try:
            if throttle_s:
                time.sleep(throttle_s)
            matrix = fetch_options_matrix(sym, d, base_url=base_url)
            with out_path.open("w") as f:
                json.dump(matrix, f)
            written += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{d}: {exc}")

    manifest = {
        "symbol": sym,
        "dates": dates,
        "n_dates": len(dates),
        "written": written,
        "skipped": skipped,
        "errors": errors[:50],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "alpaca-options-matrix-backup",
        "worker_url": base_url,
    }
    man_path = root / "manifests" / f"{sym}.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with man_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default="", help="Comma-separated tickers (default: discover from KV)")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_TRADING_DAYS)
    parser.add_argument("--from", dest="date_from", default=None)
    parser.add_argument("--to", dest="date_to", default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--throttle", type=float, default=0.05)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--limit-symbols", type=int, default=0)
    args = parser.parse_args()

    root = resolve_matrix_root(args.output_dir)
    base_url = (args.base_url or alpaca_backup_worker_url()).rstrip("/")
    print(f"Output root: {root}")
    print(f"Source worker: {base_url}")
    h = health(base_url)
    print(f"Health: {h}")

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list_symbols(base_url)
        if not symbols:
            # sensible default smoke universe
            symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "NFLX"]
            print(f"Manifest empty; using default universe ({len(symbols)} symbols)")

    if args.limit_symbols > 0:
        symbols = symbols[: args.limit_symbols]

    index = {
        "name": "Gamma Squeeze Matrix",
        "source_kv": "alpaca-options-matrix-backup",
        "worker_url": base_url,
        "symbols": symbols,
        "lookback_days": args.lookback_days,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
    }
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed": [],
        "failed": [],
    }

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym} …")
        try:
            man = export_symbol(
                sym,
                root=root,
                base_url=base_url,
                lookback_days=args.lookback_days,
                date_from=args.date_from,
                date_to=args.date_to,
                resume=args.resume,
                throttle_s=args.throttle,
            )
            state["completed"].append(
                {"symbol": sym, "written": man["written"], "skipped": man["skipped"], "n_dates": man["n_dates"]}
            )
            print(f"  wrote={man['written']} skipped={man['skipped']} dates={man['n_dates']} errors={len(man['errors'])}")
        except Exception as exc:  # noqa: BLE001
            state["failed"].append({"symbol": sym, "error": str(exc)})
            print(f"  FAILED: {exc}")

    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    with (root / "index.json").open("w") as f:
        json.dump(index, f, indent=2)
    with (root / "state.json").open("w") as f:
        json.dump(state, f, indent=2)
    print(f"Done. index → {root / 'index.json'}")
    return 0 if not state["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
