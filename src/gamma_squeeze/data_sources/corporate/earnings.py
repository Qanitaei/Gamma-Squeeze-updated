"""Earnings calendar / results adapter."""

from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request
from typing import Any

from gamma_squeeze.data_sources.base import not_configured, ok
from gamma_squeeze.data_sources.registry import register

try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()


class EarningsAdapter:
    name = "corporate.earnings"
    category = "corporate"

    def health(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "configured": bool(os.getenv("POLYGON_API_KEY") or os.getenv("FINNHUB_API_KEY")),
        }

    def fetch(self, symbol: str) -> dict:
        sym = symbol.upper()
        key = os.getenv("POLYGON_API_KEY", "").strip()
        if key:
            url = f"https://api.polygon.io/vX/reference/financials?ticker={sym}&limit=5&apiKey={key}"
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=45, context=_SSL) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                results = payload.get("results") or []
                return ok(
                    self.name,
                    self.category,
                    symbol=sym,
                    records=results,
                    meta={"origin": "polygon_financials"},
                ).to_dict()
            except Exception as exc:  # noqa: BLE001
                return {
                    "source": self.name,
                    "category": self.category,
                    "symbol": sym,
                    "success": False,
                    "configured": True,
                    "error": str(exc),
                    "records": [],
                    "data": {},
                    "meta": {},
                }
        finnhub = os.getenv("FINNHUB_API_KEY", "").strip()
        if finnhub:
            qs = urllib.parse.urlencode({"symbol": sym, "token": finnhub})
            url = f"https://finnhub.io/api/v1/calendar/earnings?{qs}"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=45, context=_SSL) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return ok(
                    self.name,
                    self.category,
                    symbol=sym,
                    data=payload if isinstance(payload, dict) else {"raw": payload},
                    meta={"origin": "finnhub"},
                ).to_dict()
            except Exception as exc:  # noqa: BLE001
                return {
                    "source": self.name,
                    "category": self.category,
                    "symbol": sym,
                    "success": False,
                    "configured": True,
                    "error": str(exc),
                    "records": [],
                    "data": {},
                    "meta": {},
                }
        return not_configured(
            self.name,
            self.category,
            hint="Set POLYGON_API_KEY or FINNHUB_API_KEY",
        ).to_dict()


register("corporate.earnings", EarningsAdapter)
