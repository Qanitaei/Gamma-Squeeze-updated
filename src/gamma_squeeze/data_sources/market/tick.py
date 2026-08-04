"""Tick / latest trade adapter — Alpaca Market Data (Schwab removed)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.config import alpaca_configured
from gamma_squeeze.data_sources.base import not_configured, ok
from gamma_squeeze.data_sources.registry import register


class TickDataAdapter:
    name = "market.tick"
    category = "market"

    def health(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "configured": alpaca_configured(),
            "providers": ["alpaca"],
        }

    def fetch(self, symbol: str, *, limit: int = 100) -> dict:
        if not alpaca_configured():
            return not_configured(
                self.name,
                self.category,
                hint="Set ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY for tick/trades",
            ).to_dict()
        try:
            from gamma_squeeze.ingest.alpaca_api import fetch_latest_trade

            trade = fetch_latest_trade(symbol)
            records = [trade] if trade else []
            return ok(
                self.name,
                self.category,
                symbol=symbol.upper(),
                records=records[:limit],
                meta={"origin": "alpaca_api", "limit": limit},
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            return {
                "source": self.name,
                "category": self.category,
                "symbol": symbol.upper(),
                "success": False,
                "configured": True,
                "error": str(exc),
                "records": [],
                "data": {},
                "meta": {},
            }


register("market.tick", TickDataAdapter)
