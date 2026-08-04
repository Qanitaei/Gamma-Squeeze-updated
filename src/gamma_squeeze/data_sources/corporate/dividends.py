"""Dividends adapter."""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import Any

from gamma_squeeze.data_sources.base import not_configured, ok
from gamma_squeeze.data_sources.registry import register

try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()


class DividendsAdapter:
    name = "corporate.dividends"
    category = "corporate"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": bool(os.getenv("POLYGON_API_KEY"))}

    def fetch(self, symbol: str) -> dict:
        key = os.getenv("POLYGON_API_KEY", "").strip()
        if not key:
            return not_configured(self.name, self.category, hint="Set POLYGON_API_KEY").to_dict()
        sym = symbol.upper()
        url = f"https://api.polygon.io/v3/reference/dividends?ticker={sym}&limit=20&apiKey={key}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45, context=_SSL) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return ok(
                self.name,
                self.category,
                symbol=sym,
                records=payload.get("results") or [],
                meta={"origin": "polygon"},
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


register("corporate.dividends", DividendsAdapter)
