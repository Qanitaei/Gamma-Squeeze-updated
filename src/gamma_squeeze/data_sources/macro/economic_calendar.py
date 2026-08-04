"""Economic calendar adapter (Finviz / Trading Economics worker)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.macro_client import fetch_calendar_near


class EconomicCalendarAdapter:
    name = "macro.economic_calendar"
    category = "macro"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True}

    def fetch(self, as_of: str, *, look_ahead_days: int = 14) -> dict:
        events = fetch_calendar_near(as_of, look_ahead_days=look_ahead_days)
        return ok(
            self.name,
            self.category,
            as_of=as_of,
            records=events,
            meta={"look_ahead_days": look_ahead_days, "n": len(events)},
        ).to_dict()


register("macro.economic_calendar", EconomicCalendarAdapter)
