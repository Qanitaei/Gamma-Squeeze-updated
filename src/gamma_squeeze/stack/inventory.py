"""Locked programming-stack inventory (Phase 2) with optional import probes."""

from __future__ import annotations

import importlib
import sys
from typing import Any, Final, TypedDict


class StackComponent(TypedDict):
    name: str
    category: str
    role: str
    install: str
    import_names: list[str]


# Canonical locked stack — must match docs/PROGRAMMING_STACK.md
LOCKED_STACK: Final[tuple[StackComponent, ...]] = (
    # Language
    {
        "name": "Python",
        "category": "language",
        "role": "Runtime",
        "install": "core",
        "import_names": [],
    },
    # Frameworks
    {
        "name": "PyTorch",
        "category": "frameworks",
        "role": "Deep learning runtime",
        "install": "ml",
        "import_names": ["torch"],
    },
    {
        "name": "Lightning",
        "category": "frameworks",
        "role": "Training loops / multi-GPU",
        "install": "ml",
        "import_names": ["lightning"],
    },
    {
        "name": "XGBoost",
        "category": "frameworks",
        "role": "Gradient-boosted direction / squeeze heads",
        "install": "ml",
        "import_names": ["xgboost"],
    },
    {
        "name": "LightGBM",
        "category": "frameworks",
        "role": "Fast tabular boosters",
        "install": "ml",
        "import_names": ["lightgbm"],
    },
    {
        "name": "CatBoost",
        "category": "frameworks",
        "role": "Ordered-boosting tabular models",
        "install": "ml",
        "import_names": ["catboost"],
    },
    {
        "name": "Stable-Baselines3",
        "category": "frameworks",
        "role": "PPO trading agents",
        "install": "rl",
        "import_names": ["stable_baselines3"],
    },
    {
        "name": "Ray RLlib",
        "category": "frameworks",
        "role": "Distributed RL",
        "install": "rl",
        "import_names": ["ray"],
    },
    {
        "name": "Darts",
        "category": "frameworks",
        "role": "Time-series forecasting toolkit",
        "install": "forecast",
        "import_names": ["darts"],
    },
    {
        "name": "PyTorch Forecasting",
        "category": "frameworks",
        "role": "TFT and related temporal models",
        "install": "forecast",
        "import_names": ["pytorch_forecasting"],
    },
    {
        "name": "Scikit-Learn",
        "category": "frameworks",
        "role": "Baselines, calibration, pipelines",
        "install": "core",
        "import_names": ["sklearn"],
    },
    {
        "name": "SHAP",
        "category": "frameworks",
        "role": "Model explainability",
        "install": "explain",
        "import_names": ["shap"],
    },
    {
        "name": "Optuna",
        "category": "frameworks",
        "role": "Hyperparameter search",
        "install": "train",
        "import_names": ["optuna"],
    },
    {
        "name": "MLflow",
        "category": "frameworks",
        "role": "Experiment tracking / model registry",
        "install": "train",
        "import_names": ["mlflow"],
    },
    # Backend
    {
        "name": "FastAPI",
        "category": "backend",
        "role": "Independent microservice HTTP APIs",
        "install": "core",
        "import_names": ["fastapi"],
    },
    {
        "name": "Redis",
        "category": "backend",
        "role": "Cache, pub/sub alerts, rate limits",
        "install": "core",
        "import_names": ["redis"],
    },
    {
        "name": "PostgreSQL",
        "category": "backend",
        "role": "Operational metadata, alerts, registry",
        "install": "core",
        "import_names": ["psycopg", "sqlalchemy"],
    },
    {
        "name": "DuckDB",
        "category": "backend",
        "role": "Local analytical queries over SSD / Parquet",
        "install": "core",
        "import_names": ["duckdb"],
    },
    {
        "name": "Apache Arrow",
        "category": "backend",
        "role": "Columnar interchange",
        "install": "core",
        "import_names": ["pyarrow"],
    },
    {
        "name": "Polars",
        "category": "backend",
        "role": "High-performance feature frames",
        "install": "core",
        "import_names": ["polars"],
    },
    # Streaming
    {
        "name": "Kafka",
        "category": "streaming",
        "role": "Feature / forecast / alert event bus",
        "install": "streaming",
        "import_names": ["confluent_kafka"],
    },
    {
        "name": "WebSockets",
        "category": "streaming",
        "role": "Live dashboard + alert streaming",
        "install": "core",
        "import_names": ["websockets"],
    },
    # Deployment (infra — not Python imports)
    {
        "name": "Docker",
        "category": "deployment",
        "role": "Local + CI container images",
        "install": "infra",
        "import_names": [],
    },
    {
        "name": "Kubernetes",
        "category": "deployment",
        "role": "Production service orchestration",
        "install": "infra",
        "import_names": [],
    },
    # Visualization
    {
        "name": "React",
        "category": "visualization",
        "role": "UI component model",
        "install": "web",
        "import_names": [],
    },
    {
        "name": "Next.js",
        "category": "visualization",
        "role": "Primary dashboard app",
        "install": "web",
        "import_names": [],
    },
    {
        "name": "Plotly",
        "category": "visualization",
        "role": "Research / interactive charts",
        "install": "viz",
        "import_names": ["plotly"],
    },
    {
        "name": "TradingView Lightweight Charts",
        "category": "visualization",
        "role": "Price / overlay charts",
        "install": "web",
        "import_names": [],
    },
    {
        "name": "ECharts",
        "category": "visualization",
        "role": "Regime / distribution / hedge curves",
        "install": "web",
        "import_names": [],
    },
)

CATEGORIES: Final[tuple[str, ...]] = (
    "language",
    "frameworks",
    "backend",
    "streaming",
    "deployment",
    "visualization",
)

PYTHON_REQUIRES: Final[str] = ">=3.12"


def _probe_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:  # noqa: BLE001
        return False


def locked_stack() -> dict[str, Any]:
    """Return the locked stack grouped by category (no import probes)."""
    by_cat: dict[str, list[dict[str, str]]] = {c: [] for c in CATEGORIES}
    for item in LOCKED_STACK:
        by_cat[item["category"]].append(
            {
                "name": item["name"],
                "role": item["role"],
                "install": item["install"],
            }
        )
    return {
        "phase": 2,
        "name": "programming_stack",
        "python_requires": PYTHON_REQUIRES,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_ok": sys.version_info >= (3, 12),
        "categories": by_cat,
        "components": [
            {"name": c["name"], "category": c["category"], "install": c["install"]}
            for c in LOCKED_STACK
        ],
    }


def probe_stack() -> dict[str, Any]:
    """Probe optional imports; infra/web items report presence via paths where possible."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    probes: list[dict[str, Any]] = []
    for item in LOCKED_STACK:
        available: bool | None
        detail: str
        if item["name"] == "Python":
            available = sys.version_info >= (3, 12)
            detail = f"sys.version={sys.version.split()[0]}"
        elif item["name"] == "Docker":
            available = (root / "Dockerfile").is_file() and (
                root / "docker-compose.yml"
            ).is_file()
            detail = "Dockerfile + docker-compose.yml"
        elif item["name"] == "Kubernetes":
            available = (root / "deploy" / "k8s").is_dir()
            detail = "deploy/k8s/"
        elif item["name"] in {
            "React",
            "Next.js",
            "TradingView Lightweight Charts",
            "ECharts",
        }:
            pkg = root / "apps" / "web-dashboard" / "package.json"
            available = pkg.is_file()
            detail = "apps/web-dashboard/package.json"
            if available:
                text = pkg.read_text(encoding="utf-8")
                checks = {
                    "React": '"react"',
                    "Next.js": '"next"',
                    "TradingView Lightweight Charts": '"lightweight-charts"',
                    "ECharts": '"echarts"',
                }
                key = checks[item["name"]]
                available = key in text
                detail = f"package.json contains {key}"
        elif not item["import_names"]:
            available = None
            detail = "no python probe"
        else:
            flags = {name: _probe_import(name) for name in item["import_names"]}
            available = any(flags.values())
            detail = ",".join(f"{k}={'ok' if v else 'missing'}" for k, v in flags.items())

        probes.append(
            {
                "name": item["name"],
                "category": item["category"],
                "install": item["install"],
                "available": available,
                "detail": detail,
            }
        )

    core_missing = [
        p["name"]
        for p in probes
        if p["install"] == "core" and p["available"] is False
    ]
    return {
        **locked_stack(),
        "probes": probes,
        "core_missing": core_missing,
        "core_ok": len(core_missing) == 0,
    }
