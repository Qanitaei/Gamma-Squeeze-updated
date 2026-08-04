"""Presidential and midterm political cycle calendar."""

from __future__ import annotations

from datetime import date
from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register

# US presidential elections: first Tuesday after first Monday in November, every 4 years
PRESIDENTIAL_YEARS = list(range(1980, 2041, 4))


def first_tuesday_after_first_monday(year: int, month: int = 11) -> date:
    d = date(year, month, 1)
    # first Monday
    while d.weekday() != 0:
        d = date.fromordinal(d.toordinal() + 1)
    # Tuesday after
    return date.fromordinal(d.toordinal() + 1)


class PoliticalCycleAdapter:
    name = "calendar.political_cycle"
    category = "calendar"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True}

    def fetch(self, *, as_of: str | None = None) -> dict:
        today = date.fromisoformat(as_of) if as_of else date.today()
        presidential = [
            {
                "date": first_tuesday_after_first_monday(y).isoformat(),
                "type": "presidential_election",
                "year": y,
            }
            for y in PRESIDENTIAL_YEARS
        ]
        midterm = [
            {
                "date": first_tuesday_after_first_monday(y).isoformat(),
                "type": "midterm_election",
                "year": y,
            }
            for y in range(1982, 2043, 4)
        ]
        # Year-in-cycle: 1..4 where election year = 4
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
            records=presidential + midterm,
            data={
                "year_in_presidential_cycle": year_in_cycle,
                "phase": phase,
                "next_presidential": next(
                    (r for r in presidential if r["date"] >= today.isoformat()),
                    None,
                ),
                "next_midterm": next((r for r in midterm if r["date"] >= today.isoformat()), None),
            },
        ).to_dict()


register("calendar.political_cycle", PoliticalCycleAdapter)
