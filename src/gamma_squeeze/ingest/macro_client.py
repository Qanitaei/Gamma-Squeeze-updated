"""Macro / calendar feature pulls from Fred-Economic-data KV worker + local mirrors."""

from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gamma_squeeze.config import PLATFORM_ROOT, calendar_worker_url, fred_worker_url

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def _fetch_json(url: str, *, timeout: int = 60) -> Any:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "GammaSqueezePlatform/0.1"})
    with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def local_fred_roots() -> list[Path]:
    """Candidate local mirrors of Fred-Economic-data series files."""
    roots: list[Path] = []
    env = os.getenv("EXPORT_FRED_ECONOMIC_DATA_SSD_PATH", "").strip()
    if env:
        roots.append(Path(env).expanduser())
    roots.extend(
        [
            Path("/Volumes/PortableSSD/Fred Economic data"),
            PLATFORM_ROOT.parent / "fred-data",
            PLATFORM_ROOT.parent / "Fred-Economic-Data" / "fred-data",
            PLATFORM_ROOT / "data" / "fred-economic-data",
        ]
    )
    # de-dupe while preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def list_fred_series(*, base_url: str | None = None) -> list[str]:
    base = (base_url or fred_worker_url()).rstrip("/")
    try:
        payload = _fetch_json(f"{base}/series")
        series = payload.get("series") or []
        if isinstance(series, list) and series:
            return [str(s).upper() for s in series]
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, TypeError):
        pass
    found: set[str] = set()
    for root in local_fred_roots():
        if not root.is_dir():
            continue
        for path in root.glob("*.json"):
            name = path.stem.upper()
            if name not in {"INDEX", "ALL-SERIES", "MANIFEST"}:
                found.add(name)
    return sorted(found)


_SERIES_CACHE: dict[str, dict[str, Any]] = {}


def clear_fred_series_cache() -> None:
    _SERIES_CACHE.clear()


def fetch_fred_series(
    series_id: str,
    *,
    base_url: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load one series from KV worker, then local SSD/Desktop mirrors."""
    sid = series_id.upper().strip().strip("/")
    cache_key = f"{base_url or ''}:{sid}"
    if use_cache and cache_key in _SERIES_CACHE:
        return _SERIES_CACHE[cache_key]

    base = (base_url or fred_worker_url()).rstrip("/")
    result: dict[str, Any] | None = None
    for url in (f"{base}/series/{sid}", f"{base}/v1/series/{sid}", f"{base}/{sid}"):
        try:
            payload = _fetch_json(url)
            if isinstance(payload, dict) and payload.get("success") is False:
                continue
            result = _normalize_series_payload(sid, payload)
            break
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue

    if result is None:
        for root in local_fred_roots():
            path = root / f"{sid}.json"
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                result = _normalize_series_payload(sid, raw, origin=f"local:{path}")
                break
            except (OSError, json.JSONDecodeError, ValueError):
                continue

    if result is None:
        result = {"series_id": sid, "observations": [], "records": [], "origin": "missing"}
    if use_cache:
        _SERIES_CACHE[cache_key] = result
    return result

def fetch_fred_history(
    series_id: str,
    *,
    as_of: str | None = None,
    base_url: str | None = None,
) -> list[tuple[str, float]]:
    """Return sorted (date, value) observations, optionally truncated at as_of."""
    payload = fetch_fred_series(series_id, base_url=base_url)
    obs = _observations_from_payload(payload)
    if as_of:
        cutoff = str(as_of)[:10]
        obs = [(d, v) for d, v in obs if d <= cutoff]
    return obs


def fetch_fred_value(
    series_id: str,
    *,
    as_of: str | None = None,
    base_url: str | None = None,
) -> float | None:
    hist = fetch_fred_history(series_id, as_of=as_of, base_url=base_url)
    if not hist:
        return None
    return hist[-1][1]


def fetch_macro_snapshot(
    *,
    series_ids: list[str] | None = None,
    as_of: str | None = None,
    base_url: str | None = None,
) -> dict[str, float | None]:
    """Latest (or as-of) values for common rate/vol regime series."""
    ids = series_ids or ["FEDFUNDS", "DGS10", "DGS2", "T10Y2Y"]
    out: dict[str, float | None] = {}
    for sid in ids:
        out[sid] = fetch_fred_value(sid, as_of=as_of, base_url=base_url)
    return out


def fetch_calendar_near(
    as_of: str,
    *,
    look_ahead_days: int = 10,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Upcoming high-impact events near as_of (best-effort)."""
    del look_ahead_days  # reserved for future filtering
    base = (base_url or calendar_worker_url()).rstrip("/")
    for url in (f"{base}/calendar", f"{base}/records", f"{base}/finviz"):
        try:
            payload = _fetch_json(url)
            events = _normalize_events(payload)
            return [e for e in events if e.get("date") and as_of <= str(e["date"])[:10]][:50]
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
    return []


def _normalize_series_payload(series_id: str, payload: Any, *, origin: str = "worker") -> dict[str, Any]:
    if isinstance(payload, list):
        records = payload
        origin_out = origin
    elif isinstance(payload, dict):
        records = (
            payload.get("records")
            or payload.get("observations")
            or payload.get("data")
            or payload.get("values")
            or []
        )
        origin_out = str(payload.get("origin") or origin)
        if payload.get("namespace"):
            origin_out = f"kv:{payload.get('namespace')}"
    else:
        records = []
        origin_out = origin
    obs = _observations_from_payload({"records": records})
    return {
        "series_id": series_id,
        "namespace": "Fred-Economic-data",
        "origin": origin_out,
        "total_records": len(obs),
        "records": [{"DATE": d, "VALUE": str(v)} for d, v in obs],
        "observations": [{"date": d, "value": v} for d, v in obs],
    }


def _observations_from_payload(payload: dict[str, Any] | list[Any]) -> list[tuple[str, float]]:
    if isinstance(payload, list):
        raw = payload
    else:
        raw = (
            payload.get("observations")
            or payload.get("records")
            or payload.get("data")
            or payload.get("values")
            or []
        )
    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            try:
                items.append((str(k)[:10], float(v)))
            except (TypeError, ValueError):
                continue
        return sorted(items, key=lambda kv: kv[0])

    out: list[tuple[str, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            date = str(item.get("DATE") or item.get("date") or item.get("observation_date") or "")[:10]
            val_raw = item.get("VALUE", item.get("value", item.get("v", item.get("close"))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            date = str(item[0])[:10]
            val_raw = item[1]
        else:
            continue
        if not date or val_raw in (None, "", "."):
            continue
        try:
            out.append((date, float(val_raw)))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda kv: kv[0])
    return out


def _latest_observation(payload: dict[str, Any]) -> float | None:
    obs = _observations_from_payload(payload)
    return obs[-1][1] if obs else None


def _normalize_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("events") or payload.get("records") or payload.get("data") or []
    else:
        raw = []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "date": str(item.get("date") or item.get("datetime") or "")[:10],
                "title": item.get("title") or item.get("event") or item.get("name") or "",
                "impact": item.get("impact") or item.get("importance") or "",
                "actual": item.get("actual"),
                "forecast": item.get("forecast") or item.get("consensus"),
            }
        )
    return out
