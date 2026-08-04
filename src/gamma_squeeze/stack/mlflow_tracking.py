"""MLflow experiment tracking adapter."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.stack.settings import get_stack_settings


def start_run(experiment: str, *, run_name: str | None = None, params: dict | None = None):
    try:
        import mlflow
    except ImportError:
        return None
    settings = get_stack_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=run_name)
    if params:
        mlflow.log_params({k: str(v) for k, v in params.items()})
    return run


def log_metrics(metrics: dict[str, float]) -> bool:
    try:
        import mlflow
    except ImportError:
        return False
    mlflow.log_metrics(metrics)
    return True


def end_run() -> None:
    try:
        import mlflow

        mlflow.end_run()
    except Exception:  # noqa: BLE001
        pass


def health() -> dict[str, Any]:
    try:
        import mlflow  # noqa: F401

        return {"mlflow": "installed", "tracking_uri": get_stack_settings().mlflow_tracking_uri}
    except ImportError:
        return {"mlflow": "not_installed", "tracking_uri": get_stack_settings().mlflow_tracking_uri}
