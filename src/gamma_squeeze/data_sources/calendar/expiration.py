"""Monthly / quarterly options expiration calendar."""

from __future__ import annotations

from datetime import date
from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.calendar.opex import monthly_opex_dates, third_friday
from gamma_squeeze.data_sources.registry import register


class ExpirationCalendarAdapter:
    name = "calendar.expiration"
    category = "calendar"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True}

    def fetch(self, *, year: int | None = None) -> dict:
        y = year or date.today().year
        monthly = monthly_opex_dates(y)
        # Quarterly: Mar / Jun / Sep / Dec standard equity options
        quarterly = [
            third_friday(y, m).isoformat()
            for m in (3, 6, 9, 12)
        ]
        # Triple witching = quarterly OPEX
        return ok(
            self.name,
            self.category,
            records=[
                *[{"date": d, "type": "monthly_expiration"} for d in monthly],
                *[{"date": d, "type": "quarterly_expiration"} for d in quarterly],
                *[{"date": d, "type": "triple_witching"} for d in quarterly],
            ],
            data={
                "year": y,
                "monthly": monthly,
                "quarterly": quarterly,
                "triple_witching": quarterly,
            },
        ).to_dict()


register("calendar.expiration", ExpirationCalendarAdapter)
