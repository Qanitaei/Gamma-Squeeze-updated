"""Core platform foundations — DI, config, logging, metrics, reproducibility, ports."""

from gamma_squeeze.core.container import Container, get_container, reset_container
from gamma_squeeze.core.logging import get_logger, setup_logging
from gamma_squeeze.core.metrics_collector import MetricsCollector, get_metrics
from gamma_squeeze.core.reproducibility import seed_everything, stable_hash
from gamma_squeeze.core.settings import PlatformSettings, load_platform_settings
from gamma_squeeze.core.standards import CODING_STANDARDS, PRODUCT_NORTH_STAR, standards_meta

__all__ = [
    "CODING_STANDARDS",
    "Container",
    "MetricsCollector",
    "PRODUCT_NORTH_STAR",
    "PlatformSettings",
    "get_container",
    "get_logger",
    "get_metrics",
    "load_platform_settings",
    "reset_container",
    "seed_everything",
    "setup_logging",
    "stable_hash",
    "standards_meta",
]
