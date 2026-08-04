"""Polygon.io options matrix adapter."""

from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request
from typing import Any

from gamma_squeeze.data_sources.base import not_configured, ok
from gamma_squeeze.data_sources.options._common import normalize_matrix
from gamma_squeeze.data_sources.registry import register

try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()


class PolygonOptionsAdapter:
    name = "options.polygon"
    category = "options"
    base_url = "https://api.polygon.io"

    def health(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "configured": bool(os.getenv("POLYGON_API_KEY")),
            "base_url": self.base_url,
        }

    def fetch_matrix(self, symbol: str, as_of: str | None = None) -> dict:
        key = os.getenv("POLYGON_API_KEY", "").strip()
        if not key:
            return not_configured(
                self.name,
                self.category,
                hint="Set POLYGON_API_KEY",
            ).to_dict()
        sym = symbol.upper()
        # Snapshot endpoint (latest); dated historical via aggregates/contracts later
        path = f"/v3/snapshot/options/{sym}"
        qs = urllib.parse.urlencode({"apiKey": key})
        url = f"{self.base_url}{path}?{qs}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60, context=_SSL) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            results = payload.get("results") or []
            contracts = []
            for item in results:
                details = item.get("details") or {}
                greeks = item.get("greeks") or {}
                day = item.get("day") or {}
                contracts.append(
                    {
                        "option_symbol": details.get("ticker"),
                        "put_call": details.get("contract_type"),
                        "strike_price": details.get("strike_price"),
                        "expiration_date": details.get("expiration_date"),
                        "delta": greeks.get("delta"),
                        "gamma": greeks.get("gamma"),
                        "theta": greeks.get("theta"),
                        "vega": greeks.get("vega"),
                        "open_interest": item.get("open_interest"),
                        "total_volume": day.get("volume"),
                        "close_price": day.get("close"),
                        "volatility": item.get("implied_volatility"),
                    }
                )
            matrix = normalize_matrix(
                {"symbol": sym, "contracts": contracts, "as_of_date": as_of},
                source="polygon",
                symbol=sym,
            )
            return ok(
                self.name,
                self.category,
                symbol=sym,
                as_of=as_of,
                data={"matrix": matrix},
                meta={"n_contracts": len(contracts), "origin": "polygon_snapshot"},
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


register("options.polygon", PolygonOptionsAdapter)
