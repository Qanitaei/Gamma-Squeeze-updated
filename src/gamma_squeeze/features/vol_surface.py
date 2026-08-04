"""IV skew / term-structure features from Skew-3D IV worker (best-effort)."""

from __future__ import annotations

import json
import ssl
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gamma_squeeze.config import skew_worker_url

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def _fetch_json(url: str, *, timeout: int = 60) -> Any:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "GammaSqueezePlatform/0.1"})
    with urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_skew_features(
    symbol: str,
    as_of: str | None = None,
    *,
    base_url: str | None = None,
) -> dict[str, float | None]:
    base = (base_url or skew_worker_url()).rstrip("/")
    sym = symbol.upper()
    urls = []
    if as_of:
        urls.extend(
            [
                f"{base}/v1/as_of/{as_of}/surface/{sym}",
                f"{base}/v1/{sym}/as_of/{as_of}",
                f"{base}/v1/as_of/{as_of}/ticker-bias/{sym}",
            ]
        )
    urls.extend(
        [
            f"{base}/v1/surface/{sym}",
            f"{base}/v1/{sym}/ticker-bias",
        ]
    )

    payload: Any = None
    for url in urls:
        try:
            payload = _fetch_json(url)
            break
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue

    if not isinstance(payload, dict):
        return {
            "iv_atm": None,
            "skew_25d": None,
            "term_slope": None,
            "iv_minus_hv": None,
            "call_bias": None,
        }

    return {
        "iv_atm": _num(payload, ("iv_atm", "atm_iv", "atm")),
        "skew_25d": _num(payload, ("skew_25d", "skew", "risk_reversal_25d")),
        "term_slope": _num(payload, ("term_slope", "term_structure_slope", "calendar_slope")),
        "iv_minus_hv": _num(payload, ("iv_minus_hv", "iv_hv_spread", "mispricing")),
        "call_bias": _num(payload, ("call_bias", "ticker_bias", "call_premium_bias")),
    }


def _num(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
        # nested summary
        summary = payload.get("summary") or payload.get("bias") or {}
        if isinstance(summary, dict) and key in summary:
            try:
                return float(summary[key])
            except (TypeError, ValueError):
                continue
    return None
