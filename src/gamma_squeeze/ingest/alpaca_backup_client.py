"""Client for alpaca-options-matrix-backup worker / KV annual history."""

from __future__ import annotations

import json
import re
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from gamma_squeeze.config import alpaca_backup_worker_url

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _fetch_json(url: str, *, timeout: int = 180, retries: int = 4) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        req = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; GammaSqueezePlatform/0.1)",
            },
        )
        try:
            with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError:
            raise
        except (URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}") from last_error


def health(base_url: str | None = None) -> dict[str, Any]:
    base = (base_url or alpaca_backup_worker_url()).rstrip("/")
    try:
        return _fetch_json(f"{base}/health")
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


def fetch_global_manifest(base_url: str | None = None) -> dict[str, Any]:
    base = (base_url or alpaca_backup_worker_url()).rstrip("/")
    for path in ("/manifest", "/v1/manifest", "/"):
        try:
            payload = _fetch_json(f"{base}{path}")
            if isinstance(payload, dict):
                return payload
        except Exception:  # noqa: BLE001
            continue
    return {}


def list_symbols(base_url: str | None = None) -> list[str]:
    """Discover tickers from /tickers, global manifest, or /keys listing."""
    base = (base_url or alpaca_backup_worker_url()).rstrip("/")

    try:
        payload = _fetch_json(f"{base}/tickers")
        raw = payload.get("tickers") or payload.get("symbols") or payload.get("universe") or []
        if isinstance(raw, list) and raw:
            out: list[str] = []
            for item in raw:
                if isinstance(item, dict):
                    sym = str(item.get("symbol") or item.get("ticker") or "").upper()
                else:
                    sym = str(item).upper()
                if sym and sym.replace(".", "").isalnum() and 1 <= len(sym) <= 8:
                    out.append(sym)
            if out:
                return sorted(set(out))
    except Exception:  # noqa: BLE001
        pass

    manifest = fetch_global_manifest(base)
    symbols: list[str] = []
    for key in ("symbols", "tickers", "universe"):
        raw = manifest.get(key)
        if isinstance(raw, list):
            for x in raw:
                if isinstance(x, dict):
                    sym = str(x.get("symbol") or x.get("ticker") or "").upper()
                else:
                    sym = str(x).upper()
                if sym:
                    symbols.append(sym)
    if symbols:
        return sorted(set(symbols))

    cursor = ""
    found: set[str] = set()
    for _ in range(50):
        url = f"{base}/keys?limit=1000"
        if cursor:
            url += f"&cursor={quote(cursor, safe='')}"
        try:
            payload = _fetch_json(url)
        except Exception:  # noqa: BLE001
            break
        for item in payload.get("keys") or []:
            name = item.get("name") if isinstance(item, dict) else str(item)
            if not name or "/" not in name:
                continue
            sym = name.split("/", 1)[0].upper()
            if sym.replace(".", "").isalnum() and 1 <= len(sym) <= 8:
                found.add(sym)
        if payload.get("list_complete", True):
            break
        cursor = payload.get("cursor") or ""
        if not cursor:
            break
    return sorted(found)


def list_matrix_dates_via_keys(symbol: str, base_url: str | None = None) -> list[str]:
    """List dated matrix keys via Worker /keys (full history)."""
    base = (base_url or alpaca_backup_worker_url()).rstrip("/")
    sym = symbol.upper()
    prefix = quote(f"{sym}/options_matrix/", safe="")
    url = f"{base}/keys?prefix={prefix}&limit=1000"
    try:
        payload = _fetch_json(url)
    except Exception:  # noqa: BLE001
        return []
    raw_keys = payload.get("keys") or []
    dates: list[str] = []
    for item in raw_keys:
        name = item.get("name") if isinstance(item, dict) else str(item)
        if not name or "/part/" in name or name.endswith("/latest"):
            continue
        m = _DATE_RE.search(name)
        if m:
            dates.append(m.group(1))
    return sorted(set(dates))


def list_matrix_dates(symbol: str, base_url: str | None = None) -> list[str]:
    """Prefer Keys listing (full history); fall back to ticker dates / manifest."""
    base = (base_url or alpaca_backup_worker_url()).rstrip("/")
    via_keys = list_matrix_dates_via_keys(symbol, base_url=base)
    if via_keys:
        return via_keys

    try:
        dates_payload = _fetch_json(f"{base}/v1/{symbol.upper()}/dates")
        dates = dates_payload.get("dates") or []
        if dates:
            return sorted({str(d)[:10] for d in dates if d})
    except Exception:  # noqa: BLE001
        pass

    try:
        manifest = _fetch_json(f"{base}/v1/{symbol.upper()}")
    except Exception:  # noqa: BLE001
        return []
    dates = list(manifest.get("dates") or [])
    for snap in manifest.get("snapshots") or []:
        d = snap.get("export_date") or snap.get("date_obtained")
        if d:
            dates.append(str(d)[:10])
    latest = manifest.get("latest_date") or manifest.get("as_of_date")
    if latest:
        dates.append(str(latest)[:10])
    return sorted({str(d)[:10] for d in dates if d})


def fetch_options_matrix(
    symbol: str,
    date: str | None = None,
    *,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Load a dated (or latest) options matrix payload suitable for GEX/DEX."""
    sym = symbol.upper()
    base = (base_url or alpaca_backup_worker_url()).rstrip("/")
    if date:
        urls = [
            f"{base}/v1/{sym}/as_of/{date}",
            f"{base}/v1/{sym}/options_matrix/{date}",
        ]
    else:
        urls = [
            f"{base}/v1/{sym}/as_of/latest",
            f"{base}/v1/{sym}/options_matrix/latest",
        ]

    last_err: Exception | None = None
    payload: Any = None
    for url in urls:
        try:
            payload = _fetch_json(url)
            break
        except HTTPError as exc:
            last_err = exc
            if exc.code == 404:
                continue
            raise RuntimeError(
                f"Options KV HTTP {exc.code} for {sym}: {exc.read().decode(errors='replace')}"
            ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_err = exc
            continue
    else:
        if isinstance(last_err, HTTPError) and last_err.code == 404:
            raise FileNotFoundError(f"Missing options matrix for {sym} date={date or 'latest'}") from last_err
        raise RuntimeError(f"Options KV fetch failed for {sym}: {last_err}") from last_err

    if isinstance(payload, dict) and payload.get("success") is False:
        raise FileNotFoundError(payload.get("error") or f"Missing matrix {sym}")

    if isinstance(payload, dict) and "contracts" not in payload and "by_expiration" not in payload:
        for key in ("data", "matrix", "records"):
            nested = payload.get(key)
            if isinstance(nested, dict) and ("contracts" in nested or "by_expiration" in nested):
                payload = nested
                break
            if isinstance(nested, list):
                payload = {"symbol": sym, "contracts": nested}

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected options payload type for {sym}: {type(payload)}")

    if not payload.get("symbol"):
        payload["symbol"] = sym
    if date and not payload.get("as_of_date"):
        payload["as_of_date"] = date
    return payload


def load_local_matrix(root: Path, symbol: str, date: str) -> dict[str, Any] | None:
    path = root / "tickers" / symbol.upper() / f"{date}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def list_local_dates(root: Path, symbol: str) -> list[str]:
    ticker_dir = root / "tickers" / symbol.upper()
    if not ticker_dir.is_dir():
        return []
    dates = [p.stem for p in ticker_dir.glob("????-??-??.json")]
    return sorted(dates)


def fetch_matrices_parallel(
    symbol: str,
    dates: list[str],
    *,
    base_url: str | None = None,
    workers: int = 8,
    throttle_s: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Download many dated matrices; returns date → matrix."""
    out: dict[str, dict[str, Any]] = {}
    base = base_url or alpaca_backup_worker_url()

    def _one(d: str) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            if throttle_s:
                time.sleep(throttle_s)
            return d, fetch_options_matrix(symbol, d, base_url=base), None
        except Exception as exc:  # noqa: BLE001
            return d, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_one, d) for d in dates]
        for fut in as_completed(futs):
            d, matrix, err = fut.result()
            if matrix is not None:
                out[d] = matrix
            elif err:
                pass
    return out
