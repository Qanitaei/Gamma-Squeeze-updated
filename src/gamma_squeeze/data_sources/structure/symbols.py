"""Per-symbol market-structure adapters (VIX, VVIX, MOVE, DXY, USDJPY, Treasury Yields)."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.data_sources.structure.fx import FXStructureAdapter
from gamma_squeeze.data_sources.structure.vol_indices import VolIndicesAdapter
from gamma_squeeze.data_sources.structure.yields import YieldsStructureAdapter


def _vol_slice(label: str) -> dict[str, Any]:
    raw = VolIndicesAdapter().fetch()
    indices = (raw.get("data") or {}).get("indices") or {}
    return {
        "symbol": label,
        "quote": indices.get(label),
        "indices": indices,
        "via": "structure.vol_indices",
    }


def _fx_slice(label: str) -> dict[str, Any]:
    raw = FXStructureAdapter().fetch()
    fx = (raw.get("data") or {}).get("fx") or {}
    return {
        "symbol": label,
        "quote": fx.get(label),
        "fx": fx,
        "via": "structure.fx",
    }


class _SymbolStructureAdapter:
    category = "structure"

    def __init__(self, name: str, label: str, kind: str) -> None:
        self.name = name
        self.label = label
        self.kind = kind

    def health(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "configured": True,
            "symbol": self.label,
            "kind": self.kind,
        }

    def fetch(self) -> dict:
        if self.kind == "vol":
            data = _vol_slice(self.label)
        elif self.kind == "fx":
            data = _fx_slice(self.label)
        else:
            raw = YieldsStructureAdapter().fetch()
            data = {
                "symbol": self.label,
                "yields": raw.get("data") or {},
                "via": "structure.yields",
            }
        return ok(self.name, self.category, data=data).to_dict()


def _factory(name: str, label: str, kind: str):
    def _make() -> _SymbolStructureAdapter:
        return _SymbolStructureAdapter(name, label, kind)

    return _make


register("structure.vix", _factory("structure.vix", "VIX", "vol"))
register("structure.vvix", _factory("structure.vvix", "VVIX", "vol"))
register("structure.move", _factory("structure.move", "MOVE", "vol"))
register("structure.dxy", _factory("structure.dxy", "DXY", "fx"))
register("structure.usdjpy", _factory("structure.usdjpy", "USDJPY", "fx"))
register(
    "structure.treasury_yields",
    _factory("structure.treasury_yields", "Treasury Yields", "yields"),
)
