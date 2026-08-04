"""Options expiration (OPEX) calendar — 3rd Friday monthly + weeklies helpers."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    # first Friday
    while d.weekday() != 4:
        d += timedelta(days=1)
    # third Friday = first + 14 days
    return d + timedelta(days=14)


def monthly_opex_dates(year: int) -> list[str]:
    return [third_friday(year, m).isoformat() for m in range(1, 13)]


class OPEXAdapter:
    name = "calendar.opex"
    category = "calendar"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True}

    def fetch(self, *, year: int | None = None, as_of: str | None = None) -> dict:
        y = year or (date.fromisoformat(as_of).year if as_of else date.today().year)
        dates = monthly_opex_dates(y)
        today = as_of or date.today().isoformat()
        upcoming = [d for d in dates if d >= today]
        return ok(
            self.name,
            self.category,
            as_of=today,
            records=[{"date": d, "type": "monthly_opex"} for d in dates],
            data={"year": y, "next_opex": upcoming[0] if upcoming else None, "upcoming": upcoming},
        ).to_dict()


register("calendar.opex", OPEXAdapter)
