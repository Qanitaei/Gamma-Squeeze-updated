#!/usr/bin/env python3
"""Upload PortableSSD phase-14 pipeline findings into gamma-squeeze-data KV.

Retention: never deletes prior keys. Writes latest + dated historical snapshots.
Full SqueezeForecast blobs are left to ``upload_squeeze_forecasts_kv.py`` —
this uploader only writes pipeline findings (does not overwrite full forecasts
at ``{TICKER}/latest`` when schema_version is already present).
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
MAX_VALUE = 20_000_000  # stay under CF KV 25MB limit with margin


def _resolve_scan_dir() -> Path:
    """Prefer PortableSSD; fall back to platform export root used in cloud agents."""
    candidates = [
        Path("/Volumes/PortableSSD/Gamma Squeeze Matrix/scans/phase14_pipeline"),
        ROOT / "data" / "exports" / "Gamma Squeeze Matrix" / "scans" / "phase14_pipeline",
    ]
    try:
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from gamma_squeeze.config import resolve_matrix_root

        candidates.insert(0, resolve_matrix_root() / "scans" / "phase14_pipeline")
    except Exception:  # noqa: BLE001
        pass
    for cand in candidates:
        if (cand / "latest.json").is_file():
            return cand
    return candidates[0]


def _load_env() -> None:
    from dotenv import dotenv_values

    for path in (
        Path("/Users/ruslantkach/Desktop/schwab-options-export/.env"),
        ROOT / ".env",
    ):
        if path.is_file():
            load_dotenv(path, override=False)
    schwab = Path("/Users/ruslantkach/Desktop/schwab-options-export/.env")
    if schwab.is_file():
        for k, v in (dotenv_values(schwab) or {}).items():
            if v and not (os.getenv(k) or "").strip():
                os.environ[k] = v


def _credentials() -> tuple[str, str]:
    _load_env()
    cred = Path("/Users/ruslantkach/Desktop/economic-calendar/.cloudflare-credentials.json")
    if cred.is_file():
        data = json.loads(cred.read_text(encoding="utf-8"))
        return str(data["account_id"]).strip(), str(data["api_token"]).strip()
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token:
        raise SystemExit("Missing Cloudflare credentials")
    return account, token


def _namespace_id() -> str:
    meta = ROOT / "data" / "cloudflare_gamma_squeeze_data.json"
    if meta.is_file():
        return str(json.loads(meta.read_text())["namespace_id"])
    for key in (
        "GAMMA_SQUEEZE_DATA_NAMESPACE_ID",
        "CLOUDFLARE_SQUEEZE_NAMESPACE_ID",
    ):
        val = os.getenv(key, "").strip()
        if val:
            return val
    raise SystemExit("Run deploy_gamma_squeeze_data_worker.py first (missing namespace id)")


def kv_put(account_id: str, token: str, namespace_id: str, key: str, value: str) -> None:
    if len(value.encode()) > MAX_VALUE:
        raise RuntimeError(f"Value too large for key {key}: {len(value.encode())} bytes")
    enc = urllib.parse.quote(key, safe="")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{enc}"
    )
    req = urllib.request.Request(
        url,
        data=value.encode("utf-8"),
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
                body = json.loads(resp.read().decode() or "{}")
            if body and body.get("success") is False:
                raise RuntimeError(body)
            return
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"PUT {key} -> {exc.code}: {exc.read()[:400]}") from exc
        except urllib.error.URLError:
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


def kv_get(account_id: str, token: str, namespace_id: str, key: str) -> Any | None:
    enc = urllib.parse.quote(key, safe="")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{enc}"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
            raw = resp.read().decode()
        return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


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


def main() -> int:
    account_id, token = _credentials()
    ns_id = _namespace_id()
    scan_dir = _resolve_scan_dir()
    if not scan_dir.is_dir():
        raise SystemExit(f"Missing scan export: {scan_dir}")

    latest_path = scan_dir / "latest.json"
    if not latest_path.is_file():
        raise SystemExit(f"Missing scan export: {latest_path}")
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    compact = _compact_scan(payload)
    stamp = _scan_stamp(payload)
    extraction_date = (
        stamp[:8]
        if len(stamp) >= 8 and stamp[:8].isdigit()
        else datetime.now(timezone.utc).strftime("%Y%m%d")
    )
    # Human date YYYY-MM-DD for day-index keys
    if len(stamp) >= 15 and "T" in stamp:
        day = f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"
    else:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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
        kv_put(account_id, token, ns_id, key, json.dumps(body, default=str))
        uploaded.append(key)

    # Day-level runs index (append; retain history — dates only, no T153655Z)
    day_key = f"scans/phase14_pipeline/by-date/{day}/runs"
    prior = kv_get(account_id, token, ns_id, day_key) or {"date": day, "runs": []}
    # Keep YYYY-MM-DD only; drop legacy T153655Z-style run ids
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
        json.dumps(
            {
                "success": True,
                "date": day,
                "runs": runs,
                "latest_date": day,
                "updated_at": now_iso,
            },
            default=str,
        ),
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
        body = json.dumps(slim, default=str)
        # Clear date-only key: AAPL/pipeline/2026-08-06 (no T153655Z)
        for key in (f"{sym}/pipeline/latest", f"{sym}/pipeline/{day}"):
            kv_put(account_id, token, ns_id, key, body)
            uploaded.append(key)
        n_sym += 1

        # Lightweight pipeline summary under a dedicated key (do not clobber full forecasts)
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
            json.dumps(pipeline_summary, default=str),
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
                json.dumps(pipeline_summary, default=str),
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
            "/v1/scans/phase14/{scan_id}",
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
    }
    kv_put(account_id, token, ns_id, "index", json.dumps(index, default=str))
    uploaded.append("index")

    summary = {
        "namespace_id": ns_id,
        "uploaded": len(uploaded),
        "n_symbols_pipeline": n_sym,
        "scan_id": stamp,
        "date": day,
        "retention": "historical",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
