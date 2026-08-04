from __future__ import annotations

from typing import Any

from gamma_squeeze.metrics.tracker import (
    compute_from_arrays,
    metrics_meta,
    track_symbol_metrics,
)


def meta() -> dict[str, Any]:
    return metrics_meta()


def track_symbols(
    symbols: list[str] | str,
    *,
    rebuild: bool = False,
    threshold: float = 0.5,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    return track_symbol_metrics(
        symbols,
        rebuild=rebuild,
        threshold=threshold,
        alpha=alpha,
    ).to_dict()


def track_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute metrics from explicit arrays in a JSON-friendly payload."""
    import numpy as np

    def arr(key: str):
        v = payload.get(key)
        if v is None:
            return None
        return np.asarray(v, dtype=float)

    return compute_from_arrays(
        y_true=arr("y_true"),
        proba=arr("proba"),
        y_pred=arr("y_pred"),
        y_reg_true=arr("y_reg_true"),
        y_reg_pred=arr("y_reg_pred"),
        returns=arr("returns"),
        signal=arr("signal"),
        trade_pnls=arr("trade_pnls"),
        hold_periods=arr("hold_periods"),
        threshold=float(payload.get("threshold", 0.5)),
        alpha=float(payload.get("alpha", 0.05)),
        periods_per_year=float(payload.get("periods_per_year", 252.0)),
        symbols=list(payload.get("symbols") or []),
    ).to_dict()
