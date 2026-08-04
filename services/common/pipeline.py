"""Canonical microservice pipeline DAG (Phase 1 — System Architecture)."""

from __future__ import annotations

from typing import Final

# Locked core pipeline: Data Collection → Trade Alert API
CORE_PIPELINE_STEPS: Final[tuple[str, ...]] = (
    "data_collection",
    "feature_engineering",
    "pattern_recognition",
    "regime_hmm",
    "xgboost_direction",
    "tft_forecasting",
    "dealer_hedging",
    "gamma_squeeze_engine",
    "ppo_agent",
    "portfolio_hedging",
    "dashboard",
    "trade_alerts",
)

# Architecture companions (later phases); optional in DAG runs
COMPANION_STEPS: Final[tuple[str, ...]] = (
    "explainability",
    "model_training",
    "performance_metrics",
)

# Full ordered list used by in-process orchestration (core + companions)
PIPELINE_STEPS: Final[tuple[str, ...]] = CORE_PIPELINE_STEPS + COMPANION_STEPS

# Locked port map (must match services.common.registry.SERVICE_REGISTRY)
LOCKED_PORTS: Final[dict[str, int]] = {
    "orchestrator": 8000,
    "data_collection": 8001,
    "feature_engineering": 8002,
    "pattern_recognition": 8003,
    "regime_hmm": 8004,
    "xgboost_direction": 8005,
    "tft_forecasting": 8006,
    "dealer_hedging": 8007,
    "gamma_squeeze_engine": 8008,
    "ppo_agent": 8009,
    "portfolio_hedging": 8010,
    "dashboard": 8011,
    "trade_alerts": 8012,
    "explainability": 8013,
    "model_training": 8014,
    "performance_metrics": 8015,
}

# Human-readable labels for docs / OpenAPI
PIPELINE_LABELS: Final[dict[str, str]] = {
    "data_collection": "Data Collection",
    "feature_engineering": "Feature Engineering",
    "pattern_recognition": "Technical Pattern Recognition",
    "regime_hmm": "Market Regime Detection (Hidden Markov Model)",
    "xgboost_direction": "XGBoost Direction Prediction",
    "tft_forecasting": "Temporal Fusion Transformer Forecasting",
    "dealer_hedging": "Dealer Hedging Simulation Engine",
    "gamma_squeeze_engine": "Gamma Squeeze Probability Engine",
    "ppo_agent": "Reinforcement Learning Trading Agent (PPO)",
    "portfolio_hedging": "Portfolio Hedging Engine",
    "dashboard": "Visualization Dashboard",
    "trade_alerts": "Trade Alert API",
    "explainability": "Explainability Engine",
    "model_training": "Model Training",
    "performance_metrics": "Performance Metrics",
    "orchestrator": "API Gateway / Orchestrator",
}

# Primary GET path used by HTTP DAG fan-out (symbol interpolated)
HTTP_STAGE_PATHS: Final[dict[str, str]] = {
    "data_collection": "/v1/collect/{symbol}",
    "feature_engineering": "/v1/features/{symbol}",
    "pattern_recognition": "/v1/patterns/{symbol}",
    "regime_hmm": "/v1/regime/{symbol}",
    "xgboost_direction": "/v1/direction/{symbol}",
    "tft_forecasting": "/v1/forecast/{symbol}",
    "dealer_hedging": "/v1/hedge-demand/{symbol}",
    "gamma_squeeze_engine": "/v1/squeeze/{symbol}",
    "ppo_agent": "/v1/action/{symbol}",
    "portfolio_hedging": "/v1/hedges/{symbol}",
    "dashboard": "/v1/dashboard/{symbol}",
    "trade_alerts": "/v1/alerts/{symbol}",
    "explainability": "/v1/explain/{symbol}",
    "model_training": "/v1/meta",
    "performance_metrics": "/v1/track/{symbol}",
}
