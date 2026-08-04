"""Top-of-book adapter via Alpaca latest quote (L2 SSD optional; no Schwab)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from gamma_squeeze.config import alpaca_configured
from gamma_squeeze.data_sources.base import not_configured, ok
from gamma_squeeze.data_sources.registry import register


class OrderBookAdapter:
    name = "market.order_book"
    category = "market"

    def health(self) -> dict[str, Any]:
        ssd = os.getenv("EXPORT_LEVEL12_SSD_PATH", "/Volumes/PortableSSD/L1/L2")
        return {
            "adapter": self.name,
            "configured": alpaca_configured() or Path(ssd).exists(),
            "level12_ssd": ssd,
            "providers": ["alpaca_quote", "ssd_l2"],
        }

    def fetch(self, symbol: str) -> dict:
        sym = symbol.upper()
        ssd = Path(os.getenv("EXPORT_LEVEL12_SSD_PATH", "/Volumes/PortableSSD/L1/L2"))
        candidates = [
            ssd / "tickers" / sym / "l2" / "latest.json",
            ssd / sym / "level2.json",
            ssd / sym / "l2.json",
        ]
        for path in candidates:
            if path.is_file():
                with path.open() as f:
                    book = json.load(f)
                return ok(
                    self.name,
                    self.category,
                    symbol=sym,
                    data={"book": book},
                    meta={"origin": str(path)},
                ).to_dict()

        if alpaca_configured():
            try:
                from gamma_squeeze.ingest.alpaca_api import fetch_latest_quote

                q = fetch_latest_quote(sym)
                book = {
                    "bids": [{"price": q.get("bp"), "size": q.get("bs")}],
                    "asks": [{"price": q.get("ap"), "size": q.get("as")}],
                    "quote": q,
                }
                return ok(
                    self.name,
                    self.category,
                    symbol=sym,
                    data={"book": book},
                    meta={"origin": "alpaca_quote", "depth": "top_of_book"},
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
            hint="Set ALPACA_API_KEY_ID/SECRET or export L2 to EXPORT_LEVEL12_SSD_PATH",
        ).to_dict()


register("market.order_book", OrderBookAdapter)
