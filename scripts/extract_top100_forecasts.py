#!/usr/bin/env python3
"""Extract full SqueezeForecast (AAPL reference schema) for top-100 tickers.

Each payload is stamped with ``extracted_at`` (UTC ISO datetime with time).
Writes dated + historical paths under forecasts/, then optionally uploads to KV.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "services"))

from gamma_squeeze.config import resolve_forecasts_root, resolve_matrix_root  # noqa: E402
from gamma_squeeze.scan.confirmed_squeeze import (  # noqa: E402
    load_top_market_cap,
    sync_matrices_to_ssd,
)
from gamma_squeeze.serve.export_forecasts import (  # noqa: E402
    attach_extraction_meta,
    export_forecast_dict,
    extraction_stamp,
    utc_now_iso,
)
from gamma_squeeze.serve.schemas import validate_forecast_dict  # noqa: E402


REQUIRED_ROOT_KEYS = (
    "schema_version",
    "symbol",
    "as_of",
    "horizons",
    "dealer_hedging_demand",
    "trade_recommendations",
    "portfolio_hedge_recommendations",
    "explanations",
    "model_versions",
    "extracted_at",
)


def _run_one(symbol: str, *, extracted_at: str) -> dict[str, Any]:
    from gamma_squeeze_engine.service import run_squeeze

    result = run_squeeze(symbol, persist=False, composite=True)
    forecast = result.get("forecast")
    if not forecast or result.get("error"):
        return {
            "symbol": symbol,
            "ok": False,
            "error": result.get("error") or "empty_forecast",
            "validation_errors": result.get("validation_errors") or [],
        }

    stamped = attach_extraction_meta(forecast, extracted_at=extracted_at)
    errors = validate_forecast_dict(stamped)
    missing = [k for k in REQUIRED_ROOT_KEYS if k not in stamped]
    if missing:
        errors = list(errors) + [f"missing_keys:{','.join(missing)}"]
    if errors:
        return {
            "symbol": symbol,
            "ok": False,
            "error": "validation_failed",
            "validation_errors": errors[:12],
            "as_of": stamped.get("as_of"),
        }

    paths = export_forecast_dict(stamped, extracted_at=extracted_at)
    h5 = next(
        (h for h in (stamped.get("horizons") or []) if h.get("horizon_days") == 5),
        None,
    )
    return {
        "symbol": symbol,
        "ok": True,
        "as_of": stamped["as_of"],
        "extracted_at": stamped["extracted_at"],
        "squeeze_probability": (h5 or {}).get("squeeze_probability"),
        "export_paths": {k: str(v) for k, v in paths.items()},
        "validation_errors": [],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    p.add_argument("--symbols", default="", help="Comma-separated override universe")
    p.add_argument("--refresh-universe", action="store_true")
    p.add_argument("--upload-kv", action="store_true", help="Upload to gamma-squeeze-data KV")
    p.add_argument("--no-sync-matrices", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    extracted_at = utc_now_iso()
    stamp = extraction_stamp(extracted_at)

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        universe = load_top_market_cap(top=args.top, refresh=args.refresh_universe)
        symbols = [str(u["symbol"]).upper() for u in universe]
    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]

    sync_info: dict[str, Any] = {}
    if not args.no_sync_matrices:
        print(f"syncing matrices for {len(symbols)} symbols (lookback=30)…")
        sync_info = sync_matrices_to_ssd(
            symbols,
            dest_root=resolve_matrix_root(),
            latest_only=False,
            lookback_dates=30,
        )
        print(json.dumps({"matrix_sync": sync_info}, indent=2, default=str))

    root = resolve_forecasts_root()
    batch_dir = root / "by-extraction" / stamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        row = _run_one(sym, extracted_at=extracted_at)
        rows.append(row)
        status = "OK" if row.get("ok") else f"FAIL:{row.get('error')}"
        print(f"[{i}/{len(symbols)}] {sym} {status} as_of={row.get('as_of')}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    ok = [r for r in rows if r.get("ok")]
    fail = [r for r in rows if not r.get("ok")]
    manifest = {
        "schema": "squeeze_forecast_extraction_v1",
        "extracted_at": extracted_at,
        "extraction_stamp": stamp,
        "extraction_date": extracted_at[:10],
        "n_symbols": len(symbols),
        "n_ok": len(ok),
        "n_fail": len(fail),
        "elapsed_sec": round(time.time() - t0, 2),
        "forecasts_root": str(root),
        "matrix_sync": sync_info,
        "reference_schema": "AAPL SqueezeForecast 1.0.0 (horizons 1–10 + composite meta)",
        "symbols_ok": [r["symbol"] for r in ok],
        "symbols_fail": [
            {"symbol": r["symbol"], "error": r.get("error"), "validation_errors": r.get("validation_errors")}
            for r in fail
        ],
        "rows": rows,
    }
    manifest_path = batch_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    # Also day-level index (append-friendly snapshot of this run)
    day_dir = root / "by-extraction-date" / extracted_at[:10]
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / f"{stamp}.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (day_dir / "latest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "extracted_at": extracted_at,
                "extraction_stamp": stamp,
                "n_ok": len(ok),
                "n_fail": len(fail),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )

    if args.upload_kv:
        sys.path.insert(0, str(ROOT / "scripts"))
        from upload_squeeze_forecasts_kv import upload_extraction_batch

        summary = upload_extraction_batch(stamp=stamp, root=root)
        print(json.dumps({"kv_upload": summary}, indent=2))

    return 0 if not fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
