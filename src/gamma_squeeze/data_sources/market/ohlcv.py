"""OHLCV adapter (Alpaca API first, KV worker fallback)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv


class OHLCVAdapter:
    name = "market.ohlcv"
    category = "market"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True}

    def fetch(self, symbol: str, *, timeframe: str = "1d") -> dict:
        sym = symbol.upper()
        try:
            df = fetch_daily_ohlcv(sym)
            records = [
                {
                    "date": str(idx),
                    "open": float(row.open) if hasattr(row, "open") else float(row["open"]),
                    "high": float(row.high) if hasattr(row, "high") else float(row["high"]),
                    "low": float(row.low) if hasattr(row, "low") else float(row["low"]),
                    "close": float(row.close) if hasattr(row, "close") else float(row["close"]),
                    "volume": float(row.volume) if hasattr(row, "volume") else float(row.get("volume") or 0),
                }
                for idx, row in df.iterrows()
            ]
            return ok(
                self.name,
                self.category,
                symbol=sym,
                as_of=records[-1]["date"] if records else None,
                records=records[-500:],
                meta={"timeframe": timeframe, "n": len(records)},
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
                "meta": {"timeframe": timeframe},
            }


register("market.ohlcv", OHLCVAdapter)
