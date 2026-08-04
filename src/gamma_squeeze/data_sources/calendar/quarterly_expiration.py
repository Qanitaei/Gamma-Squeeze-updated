"""Quarterly options expiration / triple-witching calendar adapter."""

from __future__ import annotations

from datetime import date
from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.calendar.opex import third_friday
from gamma_squeeze.data_sources.registry import register


class QuarterlyExpirationAdapter:
    name = "calendar.quarterly_expiration"
    category = "calendar"

    def health(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "configured": True,
            "months": [3, 6, 9, 12],
            "alias": "triple_witching",
        }

    def fetch(self, *, year: int | None = None) -> dict:
        y = year or date.today().year
        quarterly = [third_friday(y, m).isoformat() for m in (3, 6, 9, 12)]
        today = date.today().isoformat()
        upcoming = [d for d in quarterly if d >= today]
        return ok(
            self.name,
            self.category,
            records=[
                *[{"date": d, "type": "quarterly_expiration"} for d in quarterly],
                *[{"date": d, "type": "triple_witching"} for d in quarterly],
            ],
            data={
                "year": y,
                "quarterly": quarterly,
                "triple_witching": quarterly,
                "next_quarterly_expiration": upcoming[0] if upcoming else None,
            },
        ).to_dict()


register("calendar.quarterly_expiration", QuarterlyExpirationAdapter)
