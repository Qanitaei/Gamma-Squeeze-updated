"""Registry of all data-source adapters."""

from __future__ import annotations

from typing import Any, Callable

_FACTORY: dict[str, Callable[[], Any]] = {}


def register(name: str, factory: Callable[[], Any]) -> None:
    _FACTORY[name] = factory


def get_adapter(name: str):
    if name not in _FACTORY:
        raise KeyError(f"Unknown adapter: {name}. Known: {sorted(_FACTORY)}")
    return _FACTORY[name]()


def list_adapters() -> list[str]:
    return sorted(_FACTORY)


def catalog() -> dict[str, list[str]]:
    """Group adapter names by category prefix."""
    groups: dict[str, list[str]] = {
        "options": [],
        "market": [],
        "macro": [],
        "structure": [],
        "corporate": [],
        "calendar": [],
    }
    for name in list_adapters():
        cat = name.split(".", 1)[0]
        groups.setdefault(cat, []).append(name)
    return groups


def load_all_adapters() -> None:
    """Import adapter modules so they self-register."""
    from gamma_squeeze.data_sources import (  # noqa: F401
        calendar as _cal,
        corporate as _corp,
        macro as _macro,
        market as _mkt,
        options as _opt,
        structure as _struct,
    )
