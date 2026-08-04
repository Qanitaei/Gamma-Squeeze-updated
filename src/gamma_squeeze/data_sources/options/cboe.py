"""CBOE options / volatility products adapter."""

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


class CBOEOptionsAdapter:
    name = "options.cboe"
    category = "options"

    def health(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "configured": bool(os.getenv("CBOE_API_KEY")) or True,  # public delayed endpoints exist
            "hint": "Public CBOE delayed JSON for indices; set CBOE_API_KEY for licensed feeds",
        }

    def fetch_matrix(self, symbol: str, as_of: str | None = None) -> dict:
        """For equities, CBOE chain often requires licensed data — return index snapshot helpers."""
        sym = symbol.upper()
        # Public delayed quote probe for known CBOE symbols
        if sym in {"VIX", "VVIX", "SKEW"}:
            url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{sym}.json"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "GammaSqueezePlatform/0.2"})
                with urllib.request.urlopen(req, timeout=30, context=_SSL) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return ok(
                    self.name,
                    self.category,
                    symbol=sym,
                    as_of=as_of,
                    data={"matrix": payload},
                    meta={"origin": "cboe_cdn_delayed"},
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
        if not os.getenv("CBOE_API_KEY"):
            return not_configured(
                self.name,
                self.category,
                hint="Equity OPRA via CBOE needs CBOE_API_KEY; use VIX/VVIX for public delayed",
            ).to_dict()
        return ok(
            self.name,
            self.category,
            symbol=sym,
            as_of=as_of,
            data={"matrix": None},
            meta={"status": "stub", "message": "CBOE licensed equity chain not wired"},
        ).to_dict()


register("options.cboe", CBOEOptionsAdapter)
