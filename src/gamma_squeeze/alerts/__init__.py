"""Alert Engine — multi-trigger detection + notification channels."""

from gamma_squeeze.alerts.channels import CHANNELS, dispatch_alerts
from gamma_squeeze.alerts.engine import (
    ALERT_TRIGGERS,
    AlertThresholds,
    evaluate_alert_engine,
    engine_meta,
)

__all__ = [
    "ALERT_TRIGGERS",
    "CHANNELS",
    "AlertThresholds",
    "dispatch_alerts",
    "engine_meta",
    "evaluate_alert_engine",
]
