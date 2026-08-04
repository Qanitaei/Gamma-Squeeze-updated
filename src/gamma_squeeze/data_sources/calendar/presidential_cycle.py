"""US presidential election cycle calendar adapter."""

from __future__ import annotations

from datetime import date
from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.calendar.political_cycle import (
    PRESIDENTIAL_YEARS,
    first_tuesday_after_first_monday,
)
from gamma_squeeze.data_sources.registry import register


class PresidentialCycleAdapter:
    name = "calendar.presidential_cycle"
    category = "calendar"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "cycle_years": 4}

    def fetch(self, *, as_of: str | None = None) -> dict:
        today = date.fromisoformat(as_of) if as_of else date.today()
        records = [
            {
                "date": first_tuesday_after_first_monday(y).isoformat(),
                "type": "presidential_election",
                "year": y,
            }
            for y in PRESIDENTIAL_YEARS
        ]
        last_pres = max(y for y in PRESIDENTIAL_YEARS if y <= today.year)
        year_in_cycle = ((today.year - last_pres) % 4) + 1
        phase = {
            1: "post_election",
            2: "midterm",
            3: "pre_election",
            4: "election",
        }.get(year_in_cycle, "unknown")
        return ok(
            self.name,
            self.category,
            as_of=today.isoformat(),
            records=records,
            data={
                "year_in_presidential_cycle": year_in_cycle,
                "phase": phase,
                "next_presidential": next(
                    (r for r in records if r["date"] >= today.isoformat()),
                    None,
                ),
            },
        ).to_dict()


register("calendar.presidential_cycle", PresidentialCycleAdapter)
