"""VWAP adapter — derived from OHLCV or intraday bars."""

from __future__ import annotations

from typing import Any

import numpy as np

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv


class VWAPAdapter:
    name = "market.vwap"
    category = "market"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "method": "typical_price_volume"}

    def fetch(self, symbol: str, *, lookback: int = 20) -> dict:
        sym = symbol.upper()
        try:
            df = fetch_daily_ohlcv(sym).tail(lookback)
            if df.empty:
                return ok(self.name, self.category, symbol=sym, data={"vwap": None}).to_dict()
            typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
            vol = df["volume"].astype(float).clip(lower=0)
            denom = float(vol.sum())
            vwap = float((typical * vol).sum() / denom) if denom > 0 else float(typical.iloc[-1])
            return ok(
                self.name,
                self.category,
                symbol=sym,
                as_of=str(df.index[-1]),
                data={"vwap": vwap, "lookback": lookback, "last_close": float(df["close"].iloc[-1])},
                meta={"method": "daily_typical_price_volume"},
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


register("market.vwap", VWAPAdapter)
