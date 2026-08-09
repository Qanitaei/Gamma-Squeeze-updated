#!/usr/bin/env python3
"""Upload PortableSSD phase-14 pipeline findings into gamma-squeeze-data KV.

Retention: never deletes prior keys. Writes latest + dated historical snapshots.
Full SqueezeForecast blobs are left to ``upload_squeeze_forecasts_kv.py`` —
this uploader only writes pipeline findings (does not overwrite full forecasts
at ``{TICKER}/latest`` when schema_version is already present).

Date-only keys (no T153655Z suffixes):
  {TICKER}/pipeline/{YYYY-MM-DD}
  scans/phase14_pipeline/{YYYY-MM-DD}
"""

from __future__ import annotations

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
from gamma_squeeze.config import resolve_matrix_root  # noqa: E402

MAX_VALUE = 20_000_000


def _scan_dirs() -> list[Path]:
    roots = [
        resolve_matrix_root() / "scans" / "phase14_pipeline",
        Path("/Volumes/PortableSSD/Gamma Squeeze Matrix/scans/phase14_pipeline"),
        ROOT / "data" / "exports" / "Gamma Squeeze Matrix" / "scans" / "phase14_pipeline",
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for path in roots:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _compact_scan(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy per-stage blobs; keep findings + ranking."""
    return {
        "success": True,
        "generated_at": payload.get("generated_at"),
        "scan_id": payload.get("scan_id"),
        "universe": payload.get("universe"),
        "top_n": payload.get("top_n"),
        "n_scanned": payload.get("n_scanned"),
        "n_pipeline_ok": payload.get("n_pipeline_ok"),
        "pipeline_steps": payload.get("pipeline_steps"),
        "stage_fail_counts": payload.get("stage_fail_counts"),
        "top_squeeze_candidates": payload.get("top_squeeze_candidates"),
        "ranked_by_squeeze_probability": payload.get("ranked_by_squeeze_probability"),
        "with_alerts": [
            {
                "symbol": r.get("symbol"),
                "n_alerts": r.get("n_alerts"),
                "alert_types": r.get("alert_types"),
                "squeeze_probability": r.get("squeeze_probability"),
            }
            for r in (payload.get("with_alerts") or [])
        ],
        "findings": payload.get("findings"),
        "universe_tickers": payload.get("universe_tickers"),
        "export_dir": payload.get("export_dir"),
        "matrix_sync": payload.get("matrix_sync"),
        "source": "portable_ssd_phase14_pipeline",
    }


def _scan_stamp(payload: dict[str, Any]) -> str:
    """Prefer scan_id; else derived from generated_at."""
    sid = str(payload.get("scan_id") or "").strip()
    if sid:
        return sid
    gen = str(payload.get("generated_at") or "")
    if gen:
        try:
            dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        except ValueError:
            pass
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _day_from_stamp(stamp: str) -> str:
    if len(stamp) >= 8 and stamp[:8].isdigit():
        return f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main() -> int:
    ns_id = gamma_squeeze_namespace_id()
    account_id, token = cloudflare_credentials(namespace_id=ns_id)

    scan_dir = next((p for p in _scan_dirs() if (p / "latest.json").is_file()), None)
    if scan_dir is None:
        raise SystemExit(
            "Missing scan export latest.json under "
            + ", ".join(str(p) for p in _scan_dirs())
        )

    latest_path = scan_dir / "latest.json"
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    compact = _compact_scan(payload)
    stamp = _scan_stamp(payload)
    day = _day_from_stamp(stamp)

    findings = {
        "success": True,
        "generated_at": compact.get("generated_at"),
        "scan_id": stamp,
        "n": len(compact.get("findings") or []),
        "findings": compact.get("findings") or [],
        "ranked_by_squeeze_probability": compact.get("ranked_by_squeeze_probability") or [],
        "top_squeeze_candidates": compact.get("top_squeeze_candidates") or [],
    }

    uploaded: list[str] = []
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Latest pointers (upsert) + day-dated historical copies (YYYY-MM-DD only)
    for key, body in (
        ("scans/phase14_pipeline/latest", compact),
        ("scans/phase14_pipeline/findings", findings),
        (f"scans/phase14_pipeline/{day}", compact),
        (f"scans/phase14_pipeline/{day}/findings", findings),
    ):
        kv_put(account_id, token, ns_id, key, dumps_kv_json(body), max_value=MAX_VALUE)
        uploaded.append(key)

    # Day-level runs index (append; retain history — dates only, no T153655Z)
    day_key = f"scans/phase14_pipeline/by-date/{day}/runs"
    prior = kv_get(account_id, token, ns_id, day_key) or {"date": day, "runs": []}
    runs = [
        r
        for r in (prior.get("runs") or [])
        if isinstance(r, str) and len(r) == 10 and r[4] == "-" and r[7] == "-"
    ]
    if day not in runs:
        runs.append(day)
    kv_put(
        account_id,
        token,
        ns_id,
        day_key,
        dumps_kv_json(
            {
                "success": True,
                "date": day,
                "runs": runs,
                "latest_date": day,
                "updated_at": now_iso,
            },
        ),
        max_value=MAX_VALUE,
    )
    uploaded.append(day_key)

    symbols_dir = scan_dir / "symbols"
    n_sym = 0
    for path in sorted(symbols_dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"skip {path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        sym = str(data.get("symbol") or path.stem).upper()
        if not sym or not sym.replace(".", "").isalnum():
            continue
        slim = {
            "success": True,
            "symbol": sym,
            "rank": data.get("rank"),
            "market_cap": data.get("market_cap"),
            "ok": data.get("ok"),
            "findings": data.get("findings"),
            "stages": data.get("stages"),
            "scan_id": stamp,
            "pipeline_date": day,
            "exported_at": now_iso,
            "source": "phase14_pipeline",
        }
        body = dumps_kv_json(slim)
        # Clear date-only key: AAPL/pipeline/2026-08-06 (no T153655Z)
        for key in (f"{sym}/pipeline/latest", f"{sym}/pipeline/{day}"):
            kv_put(account_id, token, ns_id, key, body, max_value=MAX_VALUE)
            uploaded.append(key)
        n_sym += 1

        f = data.get("findings") or {}
        pipeline_summary = {
            "success": True,
            "symbol": sym,
            "as_of": f.get("as_of"),
            "squeeze_probability": f.get("squeeze_probability"),
            "expected_magnitude": f.get("squeeze_magnitude"),
            "expected_duration": f.get("squeeze_duration"),
            "confidence_score": f.get("squeeze_confidence"),
            "risk_rating": f.get("squeeze_risk"),
            "regime": f.get("regime"),
            "ppo_action": f.get("ppo_action"),
            "portfolio_structure": f.get("portfolio_structure"),
            "n_alerts": f.get("n_alerts"),
            "source": "phase14_pipeline",
            "scan_id": stamp,
            "pipeline_date": day,
            "exported_at": now_iso,
        }
        kv_put(
            account_id,
            token,
            ns_id,
            f"{sym}/pipeline/summary",
            dumps_kv_json(pipeline_summary),
            max_value=MAX_VALUE,
        )
        uploaded.append(f"{sym}/pipeline/summary")

        # Only seed {sym}/latest if empty / not a full SqueezeForecast
        existing = kv_get(account_id, token, ns_id, f"{sym}/latest")
        if not existing or not isinstance(existing, dict) or not existing.get("schema_version"):
            kv_put(
                account_id,
                token,
                ns_id,
                f"{sym}/latest",
                dumps_kv_json(pipeline_summary),
                max_value=MAX_VALUE,
            )
            uploaded.append(f"{sym}/latest")

    index = {
        "success": True,
        "service": "gamma-squeeze-data",
        "namespace": "gamma-squeeze-data",
        "openapi": "3.0.3",
        "generated_at": now_iso,
        "n_keys_this_upload": len(uploaded),
        "n_symbols_pipeline": n_sym,
        "scan": {
            "n_scanned": compact.get("n_scanned"),
            "n_pipeline_ok": compact.get("n_pipeline_ok"),
            "scan_id": stamp,
            "date": day,
        },
        "retention": "historical — dated scan/pipeline keys are never deleted",
        "keys_sample": uploaded[:40],
        "endpoints": [
            "/health",
            "/openapi.json",
            "/v1/scans/phase14/latest",
            "/v1/scans/phase14/findings",
            "/v1/scans/phase14/{YYYY-MM-DD}",
            "/v1/scans/phase14/by-date/{YYYY-MM-DD}/runs",
            "/v1/{symbol}/pipeline/latest",
            "/v1/{symbol}/pipeline/{YYYY-MM-DD}",
            "/v1/{symbol}/pipeline/summary",
            "/v1/{symbol}/forecast/latest",
            "/v1/{symbol}/forecast/{YYYY-MM-DD}",
            "/v1/extractions/{YYYY-MM-DD}",
            "/v1/extractions/{YYYY-MM-DD}/runs",
            "/v1/forecasts/index",
        ],
        "kv_key_format": {
            "forecast_dated": "{TICKER}/forecast/{YYYY-MM-DD}",
            "pipeline_dated": "{TICKER}/pipeline/{YYYY-MM-DD}",
            "example_forecast": "AAPL/forecast/2026-08-06",
            "example_pipeline": "AAPL/pipeline/2026-08-06",
        },
    }
    kv_put(account_id, token, ns_id, "index", dumps_kv_json(index), max_value=MAX_VALUE)
    uploaded.append("index")

    summary = {
        "namespace_id": ns_id,
        "uploaded": len(uploaded),
        "n_symbols_pipeline": n_sym,
        "scan_id": stamp,
        "date": day,
        "scan_dir": str(scan_dir),
        "retention": "historical",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
