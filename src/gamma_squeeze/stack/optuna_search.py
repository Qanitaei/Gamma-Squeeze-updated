"""Optuna Bayesian hyperparameter search (TPE) scaffolding."""

from __future__ import annotations

from typing import Any, Callable


def run_study(
    objective: Callable[[Any], float],
    *,
    n_trials: int = 20,
    study_name: str = "gamma-squeeze",
    direction: str = "maximize",
    seed: int = 42,
    sampler: str = "tpe",
) -> dict[str, Any]:
    try:
        import optuna
        from optuna.samplers import TPESampler, RandomSampler
    except ImportError:
        return {"status": "optuna_not_installed", "best_value": None, "best_params": {}}

    if sampler == "random":
        samp: Any = RandomSampler(seed=seed)
    else:
        samp = TPESampler(seed=seed)

    study = optuna.create_study(study_name=study_name, direction=direction, sampler=samp)
    study.optimize(objective, n_trials=n_trials)
    return {
        "status": "ok",
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "sampler": "TPESampler" if sampler != "random" else "RandomSampler",
        "bayesian": sampler != "random",
    }
