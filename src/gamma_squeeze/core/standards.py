"""Locked Phase 19 coding standards — institutional research & execution environment."""

from __future__ import annotations

from typing import Any


CODING_STANDARDS: tuple[str, ...] = (
    "Clean architecture with dependency injection",
    "Strong typing throughout the codebase",
    "Comprehensive unit and integration tests",
    "Asynchronous APIs where appropriate",
    "Modular design for swappable models",
    "Configuration via environment variables and YAML",
    "Structured logging and metrics collection",
    "Extensive documentation and inline comments",
    "Deterministic preprocessing and reproducible experiments",
)

PRODUCT_NORTH_STAR: str = (
    "Institutional quantitative research and execution environment capable of "
    "forecasting gamma squeezes, simulating dealer hedging, recommending optimal "
    "option structures, dynamically hedging portfolios, and providing fully "
    "explainable AI-driven forecasts through an interactive dashboard."
)


def standards_meta() -> dict[str, Any]:
    """Catalog of locked standards and where they are implemented."""
    return {
        "phase": 19,
        "service": "coding_standards",
        "standards": list(CODING_STANDARDS),
        "n_standards": len(CODING_STANDARDS),
        "product_north_star": PRODUCT_NORTH_STAR,
        "implementations": {
            "clean_architecture_di": {
                "layers": ["services/ (HTTP adapters)", "src/gamma_squeeze/ (domain)", "core/ (foundations)"],
                "di": "gamma_squeeze.core.container.Container",
                "ports": "gamma_squeeze.core.ports",
            },
            "strong_typing": {
                "python_requires": ">=3.12",
                "py_typed": "src/gamma_squeeze/py.typed",
                "settings": "PlatformSettings (frozen dataclass)",
                "protocols": ["PredictorPort", "TrainablePort", "ExplainerPort", "HedgerPort", "FeatureBuilderPort"],
            },
            "tests": {
                "runner": "pytest",
                "markers": ["unit", "integration"],
                "paths": ["tests/"],
            },
            "async_apis": {
                "framework": "FastAPI",
                "helpers": "gamma_squeeze.core.async_http",
                "example": "GET /v1/services/health/async",
            },
            "modular_models": {
                "registry": "gamma_squeeze.core.model_registry",
                "config_key": "models.* in config/platform.yaml",
            },
            "configuration": {
                "yaml": "config/platform.yaml",
                "env_overlay": "PLATFORM_CONFIG_PATH, GLOBAL_SEED, LOG_LEVEL, …",
                "loader": "gamma_squeeze.core.settings.load_platform_settings",
            },
            "logging_metrics": {
                "logging": "gamma_squeeze.core.logging (JSON)",
                "metrics": "gamma_squeeze.core.metrics_collector",
                "endpoint": "GET /metrics",
            },
            "documentation": {
                "docs": "docs/",
                "openapi": "/docs",
                "standards_doc": "docs/CODING_STANDARDS.md",
            },
            "reproducibility": {
                "seed": "seed_everything",
                "fingerprint": "stable_hash",
                "preprocess": "sorted_feature_matrix",
                "yaml": "reproducibility.global_seed",
            },
        },
    }
