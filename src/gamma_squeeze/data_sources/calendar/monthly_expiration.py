"""Monthly options expiration calendar adapter."""

from __future__ import annotations

from datetime import date
from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.calendar.opex import monthly_opex_dates
from gamma_squeeze.data_sources.registry import register


class MonthlyExpirationAdapter:
    name = "calendar.monthly_expiration"
    category = "calendar"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "rule": "3rd Friday monthly"}

    def fetch(self, *, year: int | None = None) -> dict:
        y = year or date.today().year
        monthly = monthly_opex_dates(y)
        today = date.today().isoformat()
        upcoming = [d for d in monthly if d >= today]
        return ok(
            self.name,
            self.category,
            records=[{"date": d, "type": "monthly_expiration"} for d in monthly],
            data={
                "year": y,
                "monthly": monthly,
                "next_monthly_expiration": upcoming[0] if upcoming else None,
            },
        ).to_dict()


register("calendar.monthly_expiration", MonthlyExpirationAdapter)
