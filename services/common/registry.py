"""Service discovery registry (local defaults; override via env)."""

from __future__ import annotations

import os
from typing import TypedDict


class ServiceInfo(TypedDict):
    name: str
    port: int
    description: str


SERVICE_REGISTRY: dict[str, ServiceInfo] = {
    "orchestrator": {
        "name": "orchestrator",
        "port": 8000,
        "description": "API gateway and pipeline orchestrator",
    },
    "data_collection": {
        "name": "data_collection",
        "port": 8001,
        "description": "Data collection from KV/SSD/macro sources",
    },
    "feature_engineering": {
        "name": "feature_engineering",
        "port": 8002,
        "description": "Options metrics, technicals, macro features",
    },
    "pattern_recognition": {
        "name": "pattern_recognition",
        "port": 8003,
        "description": "Technical pattern recognition",
    },
    "regime_hmm": {
        "name": "regime_hmm",
        "port": 8004,
        "description": "Hidden Markov Model market regime detection",
    },
    "xgboost_direction": {
        "name": "xgboost_direction",
        "port": 8005,
        "description": "XGBoost direction prediction",
    },
    "tft_forecasting": {
        "name": "tft_forecasting",
        "port": 8006,
        "description": "Temporal Fusion Transformer forecasting",
    },
    "dealer_hedging": {
        "name": "dealer_hedging",
        "port": 8007,
        "description": "Dealer hedging simulation engine",
    },
    "gamma_squeeze_engine": {
        "name": "gamma_squeeze_engine",
        "port": 8008,
        "description": "Gamma squeeze probability engine",
    },
    "ppo_agent": {
        "name": "ppo_agent",
        "port": 8009,
        "description": "PPO reinforcement learning trading agent",
    },
    "portfolio_hedging": {
        "name": "portfolio_hedging",
        "port": 8010,
        "description": "Portfolio hedging engine",
    },
    "dashboard": {
        "name": "dashboard",
        "port": 8011,
        "description": "Institutional visualization dashboard (Alpaca panels + heatmaps)",
    },
    "trade_alerts": {
        "name": "trade_alerts",
        "port": 8012,
        "description": "Alert Engine (Alpaca triggers + multi-channel notify)",
    },
    "explainability": {
        "name": "explainability",
        "port": 8013,
        "description": "Explainability engine (SHAP, attention, PDP, counterfactuals)",
    },
    "model_training": {
        "name": "model_training",
        "port": 8014,
        "description": "Model training (walk-forward, Bayesian HPO, MLflow, drift, versioning)",
    },
    "performance_metrics": {
        "name": "performance_metrics",
        "port": 8015,
        "description": "Performance metrics (classif/reg/trading/tail risk)",
    },
}


def service_url(name: str, host: str | None = None) -> str:
    info = SERVICE_REGISTRY[name]
    env_key = f"SERVICE_URL_{name.upper()}"
    override = os.getenv(env_key, "").strip()
    if override:
        return override.rstrip("/")
    h = host or os.getenv("SERVICE_HOST", "127.0.0.1")
    return f"http://{h}:{info['port']}"
