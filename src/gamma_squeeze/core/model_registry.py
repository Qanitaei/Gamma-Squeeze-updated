"""Modular model registry — swap individual models without system-wide changes."""

from __future__ import annotations

from typing import Any, Callable

from gamma_squeeze.core.ports import PredictorPort

_PREDICTORS: dict[str, Callable[[], PredictorPort]] = {}
_INSTANCES: dict[str, PredictorPort] = {}


def register_predictor(key: str, factory: Callable[[], PredictorPort]) -> None:
    """Register a replaceable predictor backend under a logical key."""
    _PREDICTORS[key] = factory
    _INSTANCES.pop(key, None)


def get_predictor(key: str) -> PredictorPort:
    if key in _INSTANCES:
        return _INSTANCES[key]
    if key not in _PREDICTORS:
        raise KeyError(f"predictor not registered: {key}. Known: {sorted(_PREDICTORS)}")
    inst = _PREDICTORS[key]()
    _INSTANCES[key] = inst
    return inst


def list_predictors() -> list[str]:
    return sorted(_PREDICTORS.keys())


def clear_predictors() -> None:
    _PREDICTORS.clear()
    _INSTANCES.clear()


class CallablePredictor:
    """Adapter wrapping a plain callable as a PredictorPort."""

    def __init__(self, name: str, fn: Callable[..., dict[str, Any]]) -> None:
        self.name = name
        self._fn = fn

    def predict(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return self._fn(symbol, **kwargs)


def bootstrap_default_predictors() -> None:
    """Wire platform defaults (lazy imports keep optional deps out of import path)."""
    if _PREDICTORS:
        return

    def _squeeze() -> PredictorPort:
        from services.gamma_squeeze_engine.service import run_squeeze

        return CallablePredictor("composite_squeeze", lambda sym, **kw: run_squeeze(sym, **kw))

    def _direction() -> PredictorPort:
        from services.xgboost_direction.service import run_direction

        return CallablePredictor("xgboost_direction", lambda sym, **kw: run_direction(sym, **kw))

    def _tft() -> PredictorPort:
        from services.tft_forecasting.service import run_tft

        return CallablePredictor("tft", lambda sym, **kw: run_tft(sym, **kw))

    def _regime() -> PredictorPort:
        from services.regime_hmm.service import run_regime

        return CallablePredictor("hmm_regime", lambda sym, **kw: run_regime(sym, **kw))

    register_predictor("squeeze", _squeeze)
    register_predictor("direction", _direction)
    register_predictor("forecast", _tft)
    register_predictor("regime", _regime)
