"""Persist SqueezeForecast JSON to SSD / local exports (dated + historical)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gamma_squeeze.config import resolve_forecasts_root
from gamma_squeeze.serve.schemas import SqueezeForecast, validate_forecast_dict


_COMPACT_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{6}Z$")


def utc_now_iso() -> str:
    """UTC extraction timestamp with seconds (Z)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extraction_stamp(extracted_at: str | None = None) -> str:
    """Filesystem/KV-safe stamp: 2026-08-06T035615Z."""
    ts = (extracted_at or utc_now_iso()).strip().replace("+00:00", "Z")
    if _COMPACT_STAMP.fullmatch(ts):
        return ts
    # Already ISO with colons: 2026-08-06T03:56:15Z
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts):
        return ts.replace(":", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    except ValueError:
        return utc_now_iso().replace(":", "")


def stamp_to_iso(stamp: str) -> str:
    """2026-08-06T035615Z → 2026-08-06T03:56:15Z"""
    s = stamp.strip()
    if _COMPACT_STAMP.fullmatch(s):
        return f"{s[:11]}{s[11:13]}:{s[13:15]}:{s[15:17]}Z"
    return s


def forecast_paths(
    symbol: str,
    as_of: str,
    *,
    extracted_at: str | None = None,
    root: Path | None = None,
) -> dict[str, Path]:
    """
    Local layout (retains history):
      by-date/{as_of}/{SYMBOL}.json
      by-ticker/{SYMBOL}/latest.json
      by-ticker/{SYMBOL}/{as_of}.json
      by-extraction/{extracted_stamp}/{SYMBOL}.json
    """
    base = root or resolve_forecasts_root()
    sym = symbol.upper()
    as_of_day = str(as_of)[:10]
    stamp = extraction_stamp(extracted_at)
    return {
        "dated": base / "by-date" / as_of_day / f"{sym}.json",
        "latest": base / "by-ticker" / sym / "latest.json",
        "ticker_dated": base / "by-ticker" / sym / f"{as_of_day}.json",
        "extraction": base / "by-extraction" / stamp / f"{sym}.json",
    }


def attach_extraction_meta(
    payload: dict[str, Any],
    *,
    extracted_at: str | None = None,
) -> dict[str, Any]:
    """Ensure schema fields used by the AAPL reference forecast + extraction time."""
    out = dict(payload)
    ts = extracted_at or utc_now_iso()
    if not str(ts).endswith("Z") and "+" not in str(ts):
        ts = utc_now_iso()
    out["extracted_at"] = ts
    out.setdefault("schema_version", out.get("schema_version") or "1.0.0")
    meta = dict(out.get("meta") or {})
    meta["extracted_at"] = ts
    meta["extraction_date"] = str(ts)[:10]
    meta["extraction_stamp"] = extraction_stamp(ts)
    out["meta"] = meta
    if out.get("as_of"):
        out["as_of"] = str(out["as_of"])[:10]
    if out.get("symbol"):
        out["symbol"] = str(out["symbol"]).upper()
    return out


def export_forecast_dict(
    payload: dict[str, Any],
    *,
    root: Path | None = None,
    extracted_at: str | None = None,
) -> dict[str, Path]:
    """Validate + write all dated/historical paths. Returns written paths."""
    stamped = attach_extraction_meta(payload, extracted_at=extracted_at)
    errors = validate_forecast_dict(stamped)
    if errors:
        raise ValueError("Invalid forecast payload: " + "; ".join(errors[:8]))

    paths = forecast_paths(
        str(stamped["symbol"]),
        str(stamped["as_of"]),
        extracted_at=str(stamped["extracted_at"]),
        root=root,
    )
    from gamma_squeeze.cloudflare_kv import dumps_kv_json

    body = dumps_kv_json(stamped, indent=2)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return paths


def export_forecast(forecast: SqueezeForecast, *, root: Path | None = None) -> Path:
    payload = forecast.to_dict()
    if not payload.get("extracted_at"):
        payload["extracted_at"] = utc_now_iso()
        forecast.extracted_at = payload["extracted_at"]
    paths = export_forecast_dict(payload, root=root, extracted_at=payload.get("extracted_at"))
    return paths["dated"]


def load_latest_forecast(symbol: str, *, root: Path | None = None) -> dict | None:
    latest = (root or resolve_forecasts_root()) / "by-ticker" / symbol.upper() / "latest.json"
    if not latest.is_file():
        return None
    with latest.open() as f:
        return json.load(f)


def forecast_day_key(payload: dict[str, Any]) -> str:
    """Calendar day for KV path: YYYY-MM-DD from extracted_at (fallback as_of)."""
    extracted = str(payload.get("extracted_at") or "").strip()
    if len(extracted) >= 10 and extracted[4] == "-" and extracted[7] == "-":
        return extracted[:10]
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    ext_day = str((meta or {}).get("extraction_date") or "")[:10]
    if len(ext_day) == 10:
        return ext_day
    as_of = str(payload.get("as_of") or "")[:10]
    if len(as_of) == 10:
        return as_of
    return utc_now_iso()[:10]


def kv_keys_for_forecast(payload: dict[str, Any]) -> dict[str, str]:
    """
    Cloudflare KV keys — clear date-only forecast paths (no T061106Z suffixes):
      {TICKER}/latest
      {TICKER}/forecast/{YYYY-MM-DD}   # e.g. AAPL/forecast/2026-08-06
      extractions/{YYYY-MM-DD}/{TICKER}
    """
    sym = str(payload["symbol"]).upper()
    day = forecast_day_key(payload)
    keys = {
        "latest": f"{sym}/latest",
        "by_date": f"{sym}/forecast/{day}",
        "extraction_copy": f"extractions/{day}/{sym}",
    }
    return {k: v for k, v in keys.items() if v}
