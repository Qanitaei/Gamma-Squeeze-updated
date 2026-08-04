"""Federal Reserve policy / balance-sheet adapter (via FRED)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.macro_client import fetch_macro_snapshot


class FederalReserveAdapter:
    name = "macro.federal_reserve"
    category = "macro"

    SERIES = ["FEDFUNDS", "DFEDTARU", "DFEDTARL", "WALCL", "RRPONTSYD", "SOFR"]

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "series": self.SERIES}

    def fetch(self) -> dict:
        snap = fetch_macro_snapshot(series_ids=self.SERIES)
        return ok(
            self.name,
            self.category,
            data={
                "fed_funds": snap.get("FEDFUNDS"),
                "target_upper": snap.get("DFEDTARU"),
                "target_lower": snap.get("DFEDTARL"),
                "balance_sheet": snap.get("WALCL"),
                "rrp": snap.get("RRPONTSYD"),
                "sofr": snap.get("SOFR"),
                "raw": snap,
            },
        ).to_dict()


register("macro.federal_reserve", FederalReserveAdapter)
