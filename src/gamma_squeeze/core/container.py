"""Lightweight dependency-injection container (clean architecture)."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Container:
    """
    Simple service locator / DI registry.

    Prefer constructor injection at call sites; use the container for
    process-wide singletons (settings, loggers, model backends).
    """

    def __init__(self) -> None:
        self._singletons: dict[str, Any] = {}
        self._factories: dict[str, Callable[[], Any]] = {}

    def register(self, key: str, instance: Any) -> None:
        self._singletons[key] = instance

    def factory(self, key: str, factory: Callable[[], Any]) -> None:
        self._factories[key] = factory

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._singletons:
            return self._singletons[key]
        if key in self._factories:
            inst = self._factories[key]()
            self._singletons[key] = inst
            return inst
        return default

    def resolve(self, key: str) -> Any:
        val = self.get(key)
        if val is None and key not in self._singletons and key not in self._factories:
            raise KeyError(f"dependency not registered: {key}")
        return val

    def has(self, key: str) -> bool:
        return key in self._singletons or key in self._factories

    def clear(self) -> None:
        self._singletons.clear()
        self._factories.clear()


_CONTAINER: Container | None = None


def get_container() -> Container:
    global _CONTAINER
    if _CONTAINER is None:
        _CONTAINER = Container()
        _bootstrap(_CONTAINER)
    return _CONTAINER


def reset_container() -> None:
    global _CONTAINER
    if _CONTAINER is not None:
        _CONTAINER.clear()
    _CONTAINER = None


def _bootstrap(c: Container) -> None:
    from gamma_squeeze.core.logging import get_logger, setup_logging
    from gamma_squeeze.core.metrics_collector import MetricsCollector
    from gamma_squeeze.core.settings import load_platform_settings

    settings = load_platform_settings()
    setup_logging(level=settings.log_level, json_logs=settings.log_json)
    c.register("settings", settings)
    c.register("logger", get_logger("gamma_squeeze"))
    c.register("metrics", MetricsCollector(namespace=settings.metrics_namespace, enabled=settings.metrics_enabled))
    # Model backend names (swap without changing callers)
    for name, backend in settings.model_backends.items():
        c.register(f"model.{name}", backend)
