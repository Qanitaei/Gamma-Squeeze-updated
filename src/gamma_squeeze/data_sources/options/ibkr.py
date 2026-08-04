"""Interactive Brokers options matrix adapter (TWS / IB Gateway)."""

from __future__ import annotations

import os
from typing import Any

from gamma_squeeze.data_sources.base import not_configured, ok
from gamma_squeeze.data_sources.registry import register


class IBKROptionsAdapter:
    name = "options.ibkr"
    category = "options"

    def health(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "configured": bool(os.getenv("IBKR_HOST") and os.getenv("IBKR_PORT")),
            "host": os.getenv("IBKR_HOST", "127.0.0.1"),
            "port": os.getenv("IBKR_PORT", "7497"),
            "client_id": os.getenv("IBKR_CLIENT_ID", "1"),
            "hint": "Requires TWS/IB Gateway + ib_insync or ibapi",
        }

    def fetch_matrix(self, symbol: str, as_of: str | None = None) -> dict:
        if not os.getenv("IBKR_HOST"):
            return not_configured(
                self.name,
                self.category,
                hint="Set IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID and run TWS/Gateway",
            ).to_dict()
        # Placeholder for live IBKR chain pull
        return ok(
            self.name,
            self.category,
            symbol=symbol.upper(),
            as_of=as_of,
            data={"matrix": None},
            meta={"status": "stub", "message": "IBKR chain fetch not yet wired; credentials detected"},
        ).to_dict()


register("options.ibkr", IBKROptionsAdapter)
