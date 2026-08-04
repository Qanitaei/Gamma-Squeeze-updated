#!/usr/bin/env python3
"""Upload PortableSSD phase-14 pipeline findings into gamma-squeeze-data KV."""

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
SSD_SCAN = Path("/Volumes/PortableSSD/Gamma Squeeze Matrix/scans/phase14_pipeline")
FORECAST_ROOTS = [
    Path("/Volumes/PortableSSD/Gamma Squeeze Matrix/forecasts"),
    ROOT / "data" / "exports" / "Gamma Squeeze Matrix" / "forecasts",
]
MAX_VALUE = 20_000_000  # stay under CF KV 25MB limit with margin


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


def main() -> int:
    account_id, token = _credentials()
    ns_id = _namespace_id()
    if not SSD_SCAN.is_dir():
        raise SystemExit(f"Missing scan export: {SSD_SCAN}")

    latest_path = SSD_SCAN / "latest.json"
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    compact = _compact_scan(payload)
    findings = {
        "success": True,
        "generated_at": compact.get("generated_at"),
        "n": len(compact.get("findings") or []),
        "findings": compact.get("findings") or [],
        "ranked_by_squeeze_probability": compact.get("ranked_by_squeeze_probability") or [],
        "top_squeeze_candidates": compact.get("top_squeeze_candidates") or [],
    }

    uploaded: list[str] = []
    kv_put(account_id, token, ns_id, "scans/phase14_pipeline/latest", json.dumps(compact, default=str))
    uploaded.append("scans/phase14_pipeline/latest")
    kv_put(account_id, token, ns_id, "scans/phase14_pipeline/findings", json.dumps(findings, default=str))
    uploaded.append("scans/phase14_pipeline/findings")

    symbols_dir = SSD_SCAN / "symbols"
    n_sym = 0
    for path in sorted(symbols_dir.glob("*.json")):
        # Skip AppleDouble / junk sidecars on PortableSSD
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
        # Prefer findings-only payload for KV size
        slim = {
            "success": True,
            "symbol": sym,
            "rank": data.get("rank"),
            "market_cap": data.get("market_cap"),
            "ok": data.get("ok"),
            "findings": data.get("findings"),
            "stages": data.get("stages"),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        key = f"{sym}/pipeline/latest"
        kv_put(account_id, token, ns_id, key, json.dumps(slim, default=str))
        uploaded.append(key)
        n_sym += 1

        # Also mirror a lightweight forecast-shaped latest from findings
        f = data.get("findings") or {}
        forecast = {
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
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        kv_put(account_id, token, ns_id, f"{sym}/latest", json.dumps(forecast, default=str))
        uploaded.append(f"{sym}/latest")

    # Optional: upload any dated forecasts already on SSD
    n_fc = 0
    for root in FORECAST_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.name in ("index.json", "manifest.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            sym = str(data.get("symbol") or path.parent.name).upper()
            as_of = str(data.get("as_of") or data.get("date") or "")[:10]
            if not sym:
                continue
            if as_of and len(as_of) == 10:
                key = f"{sym}/forecast/{as_of}"
                kv_put(account_id, token, ns_id, key, json.dumps(data, default=str))
                uploaded.append(key)
                n_fc += 1

    index = {
        "success": True,
        "service": "gamma-squeeze-data",
        "namespace": "gamma-squeeze-data",
        "openapi": "3.0.3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_keys": len(uploaded),
        "n_symbols_pipeline": n_sym,
        "n_forecast_dated": n_fc,
        "scan": {
            "n_scanned": compact.get("n_scanned"),
            "n_pipeline_ok": compact.get("n_pipeline_ok"),
            "scan_id": compact.get("scan_id"),
        },
        "keys_sample": uploaded[:40],
        "endpoints": [
            "/health",
            "/openapi.json",
            "/v1/scans/phase14/latest",
            "/v1/scans/phase14/findings",
            "/v1/{symbol}/pipeline/latest",
            "/v1/{symbol}/forecast/latest",
        ],
    }
    kv_put(account_id, token, ns_id, "index", json.dumps(index, default=str))
    uploaded.append("index")

    summary = {
        "namespace_id": ns_id,
        "uploaded": len(uploaded),
        "n_symbols_pipeline": n_sym,
        "n_forecast_dated": n_fc,
        "scan_id": compact.get("scan_id"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
