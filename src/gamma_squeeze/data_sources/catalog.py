"""Locked Phase 3 data-source adapter catalog."""

from __future__ import annotations

from typing import Any, Final, TypedDict


class AdapterSpec(TypedDict):
    id: str
    category: str
    label: str
    supports: list[str]


# Exact Phase-3 catalog requested by product architecture
LOCKED_ADAPTERS: Final[tuple[AdapterSpec, ...]] = (
    # Historical Options Matrix
    {
        "id": "options.alpaca",
        "category": "options",
        "label": "Alpaca",
        "supports": [
            "historical_options_matrix",
            "greeks",
            "alpaca_api",
            "alpaca-options-matrix-backup",
        ],
    },
    {
        "id": "options.ibkr",
        "category": "options",
        "label": "Interactive Brokers",
        "supports": ["historical_options_matrix"],
    },
    {
        "id": "options.opra",
        "category": "options",
        "label": "OPRA",
        "supports": ["historical_options_matrix", "greeks"],
    },
    {
        "id": "options.polygon",
        "category": "options",
        "label": "Polygon",
        "supports": ["historical_options_matrix", "greeks"],
    },
    {
        "id": "options.cboe",
        "category": "options",
        "label": "CBOE",
        "supports": ["historical_options_matrix", "vol_indices"],
    },
    # Market Data
    {
        "id": "market.ohlcv",
        "category": "market",
        "label": "OHLCV",
        "supports": ["bars"],
    },
    {
        "id": "market.tick",
        "category": "market",
        "label": "Tick Data",
        "supports": ["ticks"],
    },
    {
        "id": "market.vwap",
        "category": "market",
        "label": "VWAP",
        "supports": ["vwap"],
    },
    {
        "id": "market.order_book",
        "category": "market",
        "label": "Order Book",
        "supports": ["l1", "l2"],
    },
    {
        "id": "market.volume_profile",
        "category": "market",
        "label": "Volume Profile",
        "supports": ["volume_profile"],
    },
    # Macroeconomic
    {
        "id": "macro.fred",
        "category": "macro",
        "label": "FRED",
        "supports": ["series"],
    },
    {
        "id": "macro.treasury",
        "category": "macro",
        "label": "Treasury",
        "supports": ["yields"],
    },
    {
        "id": "macro.federal_reserve",
        "category": "macro",
        "label": "Federal Reserve",
        "supports": ["policy"],
    },
    {
        "id": "macro.economic_calendar",
        "category": "macro",
        "label": "Economic Calendar",
        "supports": ["events"],
    },
    # Market Structure
    {
        "id": "structure.vix",
        "category": "structure",
        "label": "VIX",
        "supports": ["vol_index"],
    },
    {
        "id": "structure.vvix",
        "category": "structure",
        "label": "VVIX",
        "supports": ["vol_index"],
    },
    {
        "id": "structure.move",
        "category": "structure",
        "label": "MOVE",
        "supports": ["vol_index"],
    },
    {
        "id": "structure.dxy",
        "category": "structure",
        "label": "DXY",
        "supports": ["fx"],
    },
    {
        "id": "structure.usdjpy",
        "category": "structure",
        "label": "USDJPY",
        "supports": ["fx"],
    },
    {
        "id": "structure.treasury_yields",
        "category": "structure",
        "label": "Treasury Yields",
        "supports": ["yields"],
    },
    # Corporate
    {
        "id": "corporate.earnings",
        "category": "corporate",
        "label": "Earnings",
        "supports": ["events"],
    },
    {
        "id": "corporate.dividends",
        "category": "corporate",
        "label": "Dividends",
        "supports": ["events"],
    },
    {
        "id": "corporate.splits",
        "category": "corporate",
        "label": "Splits",
        "supports": ["events"],
    },
    # Calendar
    {
        "id": "calendar.opex",
        "category": "calendar",
        "label": "OPEX",
        "supports": ["opex"],
    },
    {
        "id": "calendar.monthly_expiration",
        "category": "calendar",
        "label": "Monthly Expiration",
        "supports": ["expiration"],
    },
    {
        "id": "calendar.quarterly_expiration",
        "category": "calendar",
        "label": "Quarterly Expiration",
        "supports": ["expiration", "triple_witching"],
    },
    {
        "id": "calendar.presidential_cycle",
        "category": "calendar",
        "label": "Presidential Cycle",
        "supports": ["political"],
    },
    {
        "id": "calendar.midterm_cycle",
        "category": "calendar",
        "label": "Midterm Cycle",
        "supports": ["political"],
    },
)

# Convenience composites kept alongside the locked set
COMPOSITE_ADAPTERS: Final[tuple[str, ...]] = (
    "structure.vol_indices",
    "structure.fx",
    "structure.yields",
    "calendar.expiration",
    "calendar.political_cycle",
)

LOCKED_ADAPTER_IDS: Final[tuple[str, ...]] = tuple(a["id"] for a in LOCKED_ADAPTERS)


def locked_catalog() -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in LOCKED_ADAPTERS:
        by_category.setdefault(item["category"], []).append(dict(item))
    return {
        "phase": 3,
        "name": "data_sources",
        "adapters": [dict(a) for a in LOCKED_ADAPTERS],
        "by_category": by_category,
        "composites": list(COMPOSITE_ADAPTERS),
        "required_ids": list(LOCKED_ADAPTER_IDS),
    }
