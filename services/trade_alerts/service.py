from __future__ import annotations

from typing import Any

from gamma_squeeze.alerts.channels import dispatch_alerts
from gamma_squeeze.alerts.engine import AlertThresholds, engine_meta, evaluate_alert_engine


def evaluate_alerts(
    symbol: str,
    *,
    min_probability: float = 0.55,
    min_confidence: float = 0.45,
    horizon_days: int = 5,
    notify: bool = False,
    channels: list[str] | None = None,
    use_live_alpaca: bool = True,
    **threshold_overrides: float,
) -> dict[str, Any]:
    """Backward-compatible entrypoint → Alert Engine (Alpaca-forward)."""
    th = AlertThresholds(
        min_probability=min_probability,
        min_confidence=min_confidence,
        horizon_days=horizon_days,
    )
    for key, val in threshold_overrides.items():
        if hasattr(th, key) and val is not None:
            setattr(th, key, type(getattr(th, key))(val))
    return evaluate_alert_engine(
        symbol,
        thresholds=th,
        notify=notify,
        channels=channels,
        use_live_alpaca=use_live_alpaca,
    )


def notify_alerts(symbol: str, **kwargs: Any) -> dict[str, Any]:
    kwargs["notify"] = True
    return evaluate_alerts(symbol, **kwargs)


def alerts_meta() -> dict[str, Any]:
    return engine_meta()


def dispatch_only(payload: dict[str, Any], *, channels: list[str] | None = None) -> list[dict[str, Any]]:
    return dispatch_alerts(str(payload.get("symbol") or ""), payload, channels=channels)
