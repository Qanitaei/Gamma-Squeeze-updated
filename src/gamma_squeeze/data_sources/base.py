"""Common protocols and payloads for data-source adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class AdapterResult:
    """Uniform adapter response."""

    source: str
    category: str
    symbol: str | None = None
    as_of: str | None = None
    success: bool = True
    configured: bool = True
    records: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataAdapter(Protocol):
    name: str
    category: str

    def health(self) -> dict[str, Any]: ...


def not_configured(source: str, category: str, *, hint: str) -> AdapterResult:
    return AdapterResult(
        source=source,
        category=category,
        success=False,
        configured=False,
        error=f"{source} adapter not configured",
        meta={"hint": hint},
    )


def ok(
    source: str,
    category: str,
    *,
    symbol: str | None = None,
    as_of: str | None = None,
    records: list[dict[str, Any]] | None = None,
    data: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> AdapterResult:
    return AdapterResult(
        source=source,
        category=category,
        symbol=symbol,
        as_of=as_of,
        success=True,
        configured=True,
        records=records or [],
        data=data or {},
        meta=meta or {},
    )
