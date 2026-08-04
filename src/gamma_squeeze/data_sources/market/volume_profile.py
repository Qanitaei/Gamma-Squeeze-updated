"""Volume profile adapter derived from OHLCV."""

from __future__ import annotations

from typing import Any

import numpy as np

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv


class VolumeProfileAdapter:
    name = "market.volume_profile"
    category = "market"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True}

    def fetch(self, symbol: str, *, lookback: int = 60, bins: int = 24) -> dict:
        sym = symbol.upper()
        try:
            df = fetch_daily_ohlcv(sym).tail(lookback)
            if df.empty:
                return ok(self.name, self.category, symbol=sym, records=[]).to_dict()
            low = float(df["low"].astype(float).min())
            high = float(df["high"].astype(float).max())
            if high <= low:
                high = low + 1e-6
            edges = np.linspace(low, high, bins + 1)
            profile = np.zeros(bins)
            for _, row in df.iterrows():
                # distribute volume across bar range
                c = float(row["close"])
                v = float(row.get("volume") or 0)
                idx = int(np.clip(np.digitize([c], edges)[0] - 1, 0, bins - 1))
                profile[idx] += v
            records = [
                {
                    "price_lo": float(edges[i]),
                    "price_hi": float(edges[i + 1]),
                    "volume": float(profile[i]),
                }
                for i in range(bins)
            ]
            poc_i = int(np.argmax(profile))
            return ok(
                self.name,
                self.category,
                symbol=sym,
                as_of=str(df.index[-1]),
                records=records,
                data={
                    "poc": float((edges[poc_i] + edges[poc_i + 1]) / 2),
                    "vah": None,
                    "val": None,
                    "lookback": lookback,
                },
                meta={"bins": bins},
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


register("market.volume_profile", VolumeProfileAdapter)
