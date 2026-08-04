"""Bayesian hyperparameter search (Optuna TPE) with early stopping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from gamma_squeeze.training.validation import ranking_auc, time_series_splits


@dataclass
class HyperoptResult:
    status: str
    best_value: float | None = None
    best_params: dict[str, Any] = field(default_factory=dict)
    n_trials: int = 0
    sampler: str = "TPESampler"
    early_stopped: bool = False
    pruned_trials: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bayesian_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int = 25,
    n_splits: int = 3,
    study_name: str = "gamma-squeeze-bayes",
    early_stopping_rounds: int = 8,
    patience_no_improve: int = 10,
    seed: int = 42,
) -> HyperoptResult:
    """Optuna TPE (Bayesian) search over sklearn GradientBoostingClassifier params."""
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except ImportError:
        return HyperoptResult(status="optuna_not_installed")

    from sklearn.ensemble import GradientBoostingClassifier

    if len(X) < 20 or len(np.unique(y)) < 2:
        return HyperoptResult(status="insufficient_data", best_value=0.5, best_params={})

    splits = time_series_splits(len(X), n_splits=n_splits)
    best_so_far = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    pruned = 0

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=max(3, n_trials // 5), n_warmup_steps=1)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_so_far, stale, pruned
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 40, 220),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        fold_scores: list[float] = []
        for fold_i, (tr, te) in enumerate(splits):
            model = GradientBoostingClassifier(random_state=seed, **params)
            # Early stopping via staged predictions when possible
            model.fit(X[tr], y[tr])
            if hasattr(model, "staged_predict_proba") and early_stopping_rounds > 0:
                best_fold = -1.0
                best_iter = params["n_estimators"]
                no_imp = 0
                for i, proba in enumerate(model.staged_predict_proba(X[te])):
                    score = ranking_auc(y[te], proba[:, 1] if proba.ndim == 2 else proba)
                    if score > best_fold + 1e-4:
                        best_fold = score
                        best_iter = i + 1
                        no_imp = 0
                    else:
                        no_imp += 1
                    if no_imp >= early_stopping_rounds:
                        break
                fold_scores.append(best_fold)
                trial.set_user_attr(f"early_stop_iter_fold{fold_i}", best_iter)
            else:
                proba = model.predict_proba(X[te])[:, 1]
                fold_scores.append(ranking_auc(y[te], proba))

            trial.report(float(np.mean(fold_scores)), fold_i)
            if trial.should_prune():
                pruned += 1
                raise optuna.TrialPruned()

        value = float(np.mean(fold_scores))
        history.append({"trial": trial.number, "value": value, "params": dict(params)})
        if value > best_so_far + 1e-4:
            best_so_far = value
            stale = 0
        else:
            stale += 1
        if stale >= patience_no_improve:
            study.stop()
        return value

    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    early_stopped = stale >= patience_no_improve or len(study.trials) < n_trials
    if not study.trials or study.best_trial is None:
        return HyperoptResult(status="no_trials", history=history, pruned_trials=pruned)

    return HyperoptResult(
        status="ok",
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        n_trials=len(study.trials),
        sampler="TPESampler",
        early_stopped=early_stopped,
        pruned_trials=pruned,
        history=history[-20:],
    )


def fit_best_classifier(X: np.ndarray, y: np.ndarray, params: dict[str, Any] | None = None, *, seed: int = 42):
    from sklearn.ensemble import GradientBoostingClassifier

    p = {
        "n_estimators": 120,
        "learning_rate": 0.08,
        "max_depth": 3,
        "subsample": 0.9,
        "min_samples_leaf": 5,
        **(params or {}),
    }
    model = GradientBoostingClassifier(random_state=seed, **p)
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None
    model.fit(X, y)
    return model
