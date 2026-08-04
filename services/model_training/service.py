from __future__ import annotations

from typing import Any

from gamma_squeeze.training.pipeline import run_model_training, training_meta
from gamma_squeeze.training.versioning import list_versions, rollback_version


def run_training(
    symbols: list[str] | str,
    *,
    rebuild: bool = False,
    n_trials: int = 15,
    n_splits: int = 4,
    validation_method: str = "walk_forward",
    use_mlflow: bool = True,
    auto_retrain: bool = True,
    force_retrain: bool = False,
) -> dict[str, Any]:
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",") if s.strip()]
    result = run_model_training(
        symbols,
        rebuild=rebuild,
        n_trials=n_trials,
        n_splits=n_splits,
        validation_method=validation_method,
        use_mlflow=use_mlflow,
        auto_retrain=auto_retrain,
        force_retrain=force_retrain,
    )
    return result.to_dict()


def run_drift_check(
    symbols: list[str] | str,
    *,
    rebuild: bool = False,
    n_trials: int = 3,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Light training pass focused on returning the drift + retrain decision block."""
    out = run_training(
        symbols,
        rebuild=rebuild,
        n_trials=n_trials,
        n_splits=n_splits,
        validation_method="walk_forward",
        use_mlflow=False,
        auto_retrain=True,
        force_retrain=False,
    )
    return {
        "symbols": out.get("symbols"),
        "status": out.get("status"),
        "drift": out.get("drift"),
        "retrain_decision": out.get("retrain_decision"),
        "metrics": {
            k: (out.get("metrics") or {}).get(k)
            for k in ("n_rows", "n_features", "drift_triggered", "validation_auc_mean")
        },
    }


def meta() -> dict[str, Any]:
    return training_meta()


def versions(name: str = "squeeze_probability") -> dict[str, Any]:
    return {"name": name, "versions": list_versions(name)}


def rollback(name: str, version: str) -> dict[str, Any]:
    return rollback_version(name, version)
