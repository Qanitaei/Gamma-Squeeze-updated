"""Clean-architecture ports (Protocols) for swappable model backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PredictorPort(Protocol):
    """Replaceable prediction model (squeeze / direction / etc.)."""

    name: str

    def predict(self, symbol: str, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class TrainablePort(Protocol):
    """Optional train capability for modular model backends."""

    def train(self, symbols: list[str], **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class ExplainerPort(Protocol):
    def explain(self, symbol: str, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class HedgerPort(Protocol):
    def hedge(self, symbol: str, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class FeatureBuilderPort(Protocol):
    def build(self, symbol: str, **kwargs: Any) -> dict[str, Any]: ...
