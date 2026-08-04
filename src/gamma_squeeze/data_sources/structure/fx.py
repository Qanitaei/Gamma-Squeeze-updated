"""DXY / USDJPY structure adapters."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv


class FXStructureAdapter:
    name = "structure.fx"
    category = "structure"

    MAP = {
        "DXY": ["DXY", "DX-Y.NYB", "UUP"],
        "USDJPY": ["USDJPY", "USD/JPY", "FXY"],
    }

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "symbols": self.MAP}

    def fetch(self) -> dict:
        out: dict[str, Any] = {}
        for label, candidates in self.MAP.items():
            val = None
            origin = None
            for sym in candidates:
                try:
                    df = fetch_daily_ohlcv(sym)
                    if not df.empty:
                        val = float(df["close"].iloc[-1])
                        origin = sym
                        break
                except Exception:  # noqa: BLE001
                    continue
            out[label] = {"value": val, "origin": origin}
        return ok(self.name, self.category, data={"fx": out}).to_dict()


register("structure.fx", FXStructureAdapter)
