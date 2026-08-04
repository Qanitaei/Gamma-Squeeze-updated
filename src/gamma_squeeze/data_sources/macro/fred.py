"""FRED macroeconomic series adapter."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.macro_client import fetch_fred_series, fetch_macro_snapshot


class FREDAdapter:
    name = "macro.fred"
    category = "macro"

    DEFAULT_SERIES = [
        "FEDFUNDS",
        "CPIAUCSL",
        "PPIACO",
        "PAYEMS",
        "GACDISA066MSFRBNY",
        "A191RL1Q225SBEA",
        "UMCSENT",
        "DGS2",
        "DGS10",
        "DGS30",
        "T10Y2Y",
        "DTWEXBGS",
        "DEXJPUS",
        "VIXCLS",
        "BAA10Y",
        "BAMLC0A0CM",
        "BAMLH0A0HYM2",
        "WEI",
    ]

    def health(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "configured": True,
            "series": self.DEFAULT_SERIES,
            "namespace": "Fred-Economic-data",
        }

    def fetch_snapshot(self, series_ids: list[str] | None = None, as_of: str | None = None) -> dict:
        ids = series_ids or self.DEFAULT_SERIES
        snap = fetch_macro_snapshot(series_ids=ids, as_of=as_of)
        records = [{"series_id": k, "value": v} for k, v in snap.items()]
        return ok(self.name, self.category, records=records, data={"snapshot": snap}).to_dict()

    def fetch_series(self, series_id: str) -> dict:
        payload = fetch_fred_series(series_id)
        return ok(
            self.name,
            self.category,
            data={"series": payload},
            meta={"series_id": series_id.upper()},
        ).to_dict()


register("macro.fred", FREDAdapter)
