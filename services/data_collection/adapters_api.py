"""HTTP helpers for data-source adapters."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.data_sources.catalog import LOCKED_ADAPTER_IDS, locked_catalog
from gamma_squeeze.data_sources.registry import catalog, get_adapter, list_adapters, load_all_adapters


def ensure_loaded() -> None:
    load_all_adapters()


def adapters_catalog() -> dict[str, Any]:
    ensure_loaded()
    locked = locked_catalog()
    registered = list_adapters()
    missing = [aid for aid in LOCKED_ADAPTER_IDS if aid not in registered]
    return {
        "phase": 3,
        "adapters": registered,
        "by_category": catalog(),
        "locked": locked,
        "locked_ok": len(missing) == 0,
        "missing_locked": missing,
    }


def adapters_health() -> dict[str, Any]:
    ensure_loaded()
    out = {}
    for name in list_adapters():
        try:
            out[name] = get_adapter(name).health()
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": str(exc)}
    return {"health": out}


def fetch_adapter(name: str, *, symbol: str | None = None, as_of: str | None = None, **kwargs) -> dict[str, Any]:
    ensure_loaded()
    adapter = get_adapter(name)
    # Dispatch common method names
    if hasattr(adapter, "fetch_matrix") and symbol:
        return adapter.fetch_matrix(symbol, as_of=as_of)
    if hasattr(adapter, "list_dates") and symbol and kwargs.get("list_dates"):
        return adapter.list_dates(symbol)
    if hasattr(adapter, "fetch_snapshot"):
        return adapter.fetch_snapshot(kwargs.get("series_ids"))
    if hasattr(adapter, "fetch_series") and kwargs.get("series_id"):
        return adapter.fetch_series(kwargs["series_id"])
    if hasattr(adapter, "fetch"):
        if symbol is not None:
            # try symbol-first signatures
            try:
                return adapter.fetch(symbol, **{k: v for k, v in kwargs.items() if k != "list_dates"})
            except TypeError:
                pass
        try:
            return adapter.fetch(as_of=as_of, **{k: v for k, v in kwargs.items() if k != "list_dates"})
        except TypeError:
            return adapter.fetch()
    return {"success": False, "error": f"Adapter {name} has no fetch method", "source": name}
