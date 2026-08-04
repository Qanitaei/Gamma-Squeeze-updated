"""Data source adapters (options, market, macro, structure, corporate, calendar)."""

from gamma_squeeze.data_sources.base import AdapterResult
from gamma_squeeze.data_sources.catalog import LOCKED_ADAPTER_IDS, locked_catalog
from gamma_squeeze.data_sources.registry import catalog, get_adapter, list_adapters, load_all_adapters

__all__ = [
    "AdapterResult",
    "LOCKED_ADAPTER_IDS",
    "catalog",
    "get_adapter",
    "list_adapters",
    "load_all_adapters",
    "locked_catalog",
]
