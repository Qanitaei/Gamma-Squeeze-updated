"""US Treasury yields adapter (FRED DGS* + optional direct Treasury)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.macro_client import fetch_macro_snapshot


class TreasuryAdapter:
    name = "macro.treasury"
    category = "macro"

    SERIES = {
        "2y": "DGS2",
        "10y": "DGS10",
        "30y": "DGS30",
        "3m": "DGS3MO",
        "spread_10y2y": "T10Y2Y",
    }

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "map": self.SERIES}

    def fetch(self) -> dict:
        ids = list(self.SERIES.values())
        snap = fetch_macro_snapshot(series_ids=ids)
        curve = {tenor: snap.get(sid) for tenor, sid in self.SERIES.items()}
        return ok(
            self.name,
            self.category,
            data={"curve": curve, "raw": snap},
            meta={"units": "percent"},
        ).to_dict()


register("macro.treasury", TreasuryAdapter)
