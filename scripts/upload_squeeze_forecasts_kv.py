#!/usr/bin/env python3
"""Upload SqueezeForecast JSON into gamma-squeeze-data KV (retain history).

Writes clear date-only forecast keys (no T061106Z suffixes):
  {TICKER}/latest
  {TICKER}/forecast/{YYYY-MM-DD}   # e.g. AAPL/forecast/2026-08-06
  extractions/{YYYY-MM-DD}/{TICKER}
  extractions/{YYYY-MM-DD}/index
  extractions/{YYYY-MM-DD}/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.cloudflare_kv import (  # noqa: E402
    cloudflare_credentials,
    dumps_kv_json,
    gamma_squeeze_namespace_id,
    kv_get,
    kv_put,
)
from gamma_squeeze.config import resolve_forecasts_root  # noqa: E402
from gamma_squeeze.serve.export_forecasts import (  # noqa: E402
    extraction_stamp,
    kv_keys_for_forecast,
)

MAX_VALUE = 20_000_000


def _credentials() -> tuple[str, str]:
    return cloudflare_credentials()


def _namespace_id() -> str:
    return gamma_squeeze_namespace_id()


def upload_payload(
    account_id: str,
    token: str,
    namespace_id: str,
    payload: dict[str, Any],
) -> list[str]:
    body = dumps_kv_json(payload)
    keys = kv_keys_for_forecast(payload)
    written: list[str] = []
    for key in keys.values():
        kv_put(account_id, token, namespace_id, key, body)
        written.append(key)
    return written


def _candidate_forecast_roots(explicit: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if explicit is not None:
        roots.append(explicit)
    try:
        roots.append(resolve_forecasts_root())
    except Exception:  # noqa: BLE001
        pass
    roots.extend(
        [
            Path("/Volumes/PortableSSD/Gamma Squeeze Matrix/forecasts"),
            ROOT / "data" / "exports" / "Gamma Squeeze Matrix" / "forecasts",
        ]
    )
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def upload_extraction_batch(
    *,
    stamp: str | None = None,
    root: Path | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Upload one by-extraction/{stamp} batch; retain all historical keys."""
    account_id, token = _credentials()
    ns_id = _namespace_id()

    batch_dir: Path | None = None
    base: Path | None = None
    for cand in _candidate_forecast_roots(root):
        parent = cand / "by-extraction"
        if stamp:
            trial = parent / stamp
            if trial.is_dir():
                batch_dir = trial
                base = cand
                break
        elif parent.is_dir():
            dirs = sorted([d for d in parent.iterdir() if d.is_dir()], key=lambda p: p.name)
            if dirs:
                batch_dir = dirs[-1]
                stamp = batch_dir.name
                base = cand
                break
    if batch_dir is None or stamp is None:
        raise SystemExit(f"No extractions found for stamp={stamp!r}")

    stamp = extraction_stamp(stamp)
    files = sorted(batch_dir.glob("*.json"))
    files = [f for f in files if f.name not in ("manifest.json", "index.json") and not f.name.startswith("._")]
    if symbols:
        want = {s.upper() for s in symbols}
        files = [f for f in files if f.stem.upper() in want]
    print(f"upload root={base} batch={batch_dir} n_files={len(files)}")

    uploaded_keys: list[str] = []
    symbols_ok: list[str] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"skip {path.name}: {exc}")
            continue
        if not isinstance(payload, dict) or not payload.get("symbol"):
            continue
        if not payload.get("extracted_at"):
            # Reconstruct ISO from stamp for older files
            from gamma_squeeze.serve.export_forecasts import stamp_to_iso

            payload["extracted_at"] = stamp_to_iso(stamp)
        written = upload_payload(account_id, token, ns_id, payload)
        uploaded_keys.extend(written)
        symbols_ok.append(str(payload["symbol"]).upper())
        print(f"uploaded {payload['symbol']} keys={len(written)}")

    # Compact stamp 2026-08-06T061106Z → day 2026-08-06
    extraction_date = stamp[:10] if stamp[4:5] == "-" else f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"
    extracted_at_iso = None
    if symbols_ok:
        first = batch_dir / f"{symbols_ok[0]}.json"
        if first.is_file():
            extracted_at_iso = json.loads(first.read_text()).get("extracted_at")
            if extracted_at_iso:
                extraction_date = str(extracted_at_iso)[:10]
    index = {
        "success": True,
        "service": "gamma-squeeze-data",
        "kind": "forecast_extraction",
        "extracted_at": extracted_at_iso,
        "extraction_date": extraction_date,
        "n_symbols": len(symbols_ok),
        "symbols": symbols_ok,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kv_keys_per_symbol": [
            "{TICKER}/latest",
            "{TICKER}/forecast/{YYYY-MM-DD}",
            "extractions/{YYYY-MM-DD}/{TICKER}",
        ],
        "forecast_key_example": "AAPL/forecast/2026-08-06",
        "retention": "historical by calendar day (YYYY-MM-DD only)",
    }

    kv_put(account_id, token, ns_id, f"extractions/{extraction_date}/index", dumps_kv_json(index))
    uploaded_keys.append(f"extractions/{extraction_date}/index")

    day_key = f"extractions/{extraction_date}/runs"
    prior = kv_get(account_id, token, ns_id, day_key) or {"date": extraction_date, "runs": []}
    runs = list(prior.get("runs") or [])
    if extraction_date not in runs:
        runs.append(extraction_date)
    day_payload = {
        "success": True,
        "date": extraction_date,
        "runs": runs,
        "latest_date": extraction_date,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    kv_put(account_id, token, ns_id, day_key, dumps_kv_json(day_payload))
    uploaded_keys.append(day_key)

    catalog = {
        "success": True,
        "service": "gamma-squeeze-forecasts",
        "namespace": "gamma-squeeze-data",
        "latest_extraction_date": extraction_date,
        "n_symbols": len(symbols_ok),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endpoints": [
            "/v1/{symbol}/forecast/latest",
            "/v1/{symbol}/forecast/{YYYY-MM-DD}",
            "/v1/extractions/{YYYY-MM-DD}",
            "/v1/extractions/{YYYY-MM-DD}/runs",
        ],
        "kv_key_example": "AAPL/forecast/2026-08-06",
    }
    kv_put(account_id, token, ns_id, "forecasts/index", dumps_kv_json(catalog))
    uploaded_keys.append("forecasts/index")

    return {
        "namespace_id": ns_id,
        "extraction_date": extraction_date,
        "n_symbols": len(symbols_ok),
        "n_keys": len(uploaded_keys),
        "forecast_key_format": "{TICKER}/forecast/{YYYY-MM-DD}",
        "retention": "historical",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stamp", default="", help="Extraction stamp e.g. 2026-08-06T035615Z")
    p.add_argument("--latest", action="store_true", help="Upload newest by-extraction batch")
    p.add_argument("--symbols", default="", help="Optional comma filter")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    stamp = args.stamp.strip() or None
    if not stamp and not args.latest:
        args.latest = True
    summary = upload_extraction_batch(stamp=stamp, symbols=symbols)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
