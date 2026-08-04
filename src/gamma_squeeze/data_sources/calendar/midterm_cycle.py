"""US midterm election cycle calendar adapter."""

from __future__ import annotations

from datetime import date
from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.calendar.political_cycle import (
    first_tuesday_after_first_monday,
)
from gamma_squeeze.data_sources.registry import register

MIDTERM_YEARS = list(range(1982, 2043, 4))


class MidtermCycleAdapter:
    name = "calendar.midterm_cycle"
    category = "calendar"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "cycle_years": 4}

    def fetch(self, *, as_of: str | None = None) -> dict:
        today = date.fromisoformat(as_of) if as_of else date.today()
        records = [
            {
                "date": first_tuesday_after_first_monday(y).isoformat(),
                "type": "midterm_election",
                "year": y,
            }
            for y in MIDTERM_YEARS
        ]
        return ok(
            self.name,
            self.category,
            as_of=today.isoformat(),
            records=records,
            data={
                "next_midterm": next(
                    (r for r in records if r["date"] >= today.isoformat()),
                    None,
                ),
                "is_midterm_year": today.year in MIDTERM_YEARS,
            },
        ).to_dict()


register("calendar.midterm_cycle", MidtermCycleAdapter)
