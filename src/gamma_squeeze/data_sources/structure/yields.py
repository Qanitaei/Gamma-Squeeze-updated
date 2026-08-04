"""Treasury yields market-structure view (aliases macro.treasury)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.macro.treasury import TreasuryAdapter
from gamma_squeeze.data_sources.registry import register


class YieldsStructureAdapter:
    name = "structure.yields"
    category = "structure"

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "backend": "macro.treasury"}

    def fetch(self) -> dict:
        raw = TreasuryAdapter().fetch()
        return ok(
            self.name,
            self.category,
            data=raw.get("data") or {},
            meta={"via": "macro.treasury"},
        ).to_dict()


register("structure.yields", YieldsStructureAdapter)
