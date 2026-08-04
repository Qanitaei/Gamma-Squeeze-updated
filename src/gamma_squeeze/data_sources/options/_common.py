"""Shared helpers for options matrix adapters."""

from __future__ import annotations

from typing import Any


def normalize_matrix(payload: dict[str, Any], *, source: str, symbol: str) -> dict[str, Any]:
    out = dict(payload)
    out.setdefault("symbol", symbol.upper())
    out.setdefault("source", source)
    if "contracts" not in out and "by_expiration" not in out:
        for key in ("data", "matrix", "records", "options"):
            nested = out.get(key)
            if isinstance(nested, dict) and ("contracts" in nested or "by_expiration" in nested):
                out = {**nested, "symbol": symbol.upper(), "source": source}
                break
            if isinstance(nested, list):
                out = {"symbol": symbol.upper(), "source": source, "contracts": nested}
                break
    return out
