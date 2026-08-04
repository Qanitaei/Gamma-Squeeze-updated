"""Model Training pipeline — validation, Bayesian search, MLflow, versioning, drift, auto-retrain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gamma_squeeze.features.feature_store import (
    build_features_for_symbol,
    load_features,
    model_feature_matrix,
)
from gamma_squeeze.labels.label_builder import build_labels_for_symbol, load_labels
from gamma_squeeze.stack.mlflow_tracking import end_run, log_metrics, start_run
from gamma_squeeze.training.auto_retrain import AutoRetrainPolicy, decide_retrain
from gamma_squeeze.training.drift import detect_all_drift
from gamma_squeeze.training.hyperopt import bayesian_search, fit_best_classifier
from gamma_squeeze.training.validation import align_xy, evaluate_splits
from gamma_squeeze.training.versioning import list_versions, save_versioned_model


TRAINING_CAPABILITIES: tuple[str, ...] = (
    "Walk Forward Validation",
    "Time Series Split",
    "Bayesian Hyperparameter Search",
    "Early Stopping",
    "MLflow Tracking",
    "Version Control",
    "Automatic Retraining",
    "Drift Detection",
    "Feature Drift",
    "Prediction Drift",
    "Concept Drift",
)


@dataclass
class TrainingResult:
    symbols: list[str]
    status: str
    validation: dict[str, Any] = field(default_factory=dict)
    hyperopt: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    retrain_decision: dict[str, Any] = field(default_factory=dict)
    version: dict[str, Any] = field(default_factory=dict)
    mlflow: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_panels(symbols: list[str], *, rebuild: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats, labs = [], []
    for sym in symbols:
        f = build_features_for_symbol(sym, include_skew=False, include_macro=True) if rebuild else load_features(sym)
        if f.empty and not rebuild:
            f = build_features_for_symbol(sym, include_skew=False, include_macro=True)
        lab = build_labels_for_symbol(sym, features=f) if rebuild or load_labels(sym).empty else load_labels(sym)
        if not f.empty:
            feats.append(f)
        if not lab.empty:
            labs.append(lab)
    if not feats or not labs:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(feats, ignore_index=True), pd.concat(labs, ignore_index=True)


def run_model_training(
    symbols: list[str],
    *,
    rebuild: bool = False,
    n_trials: int = 15,
    n_splits: int = 4,
    validation_method: str = "walk_forward",  # walk_forward | time_series_split
    use_mlflow: bool = True,
    auto_retrain: bool = True,
    force_retrain: bool = False,
    ref_fraction: float = 0.6,
    label_col: str = "squeeze_5d",
) -> TrainingResult:
    """Full training loop with validation, Bayesian search, drift, versioning."""
    syms = [s.upper() for s in symbols]
    features, labels = _load_panels(syms, rebuild=rebuild)
    if features.empty or labels.empty:
        return TrainingResult(symbols=syms, status="no_data")

    _, cols = model_feature_matrix(features)
    X, y, merged = align_xy(features, labels, cols, label_col=label_col)
    if len(X) < 30 or len(np.unique(y)) < 2:
        return TrainingResult(
            symbols=syms,
            status="insufficient_data",
            metrics={"n_rows": int(len(X)), "label_col": label_col},
        )

    # Split reference / current for drift (temporal)
    cut = max(10, int(len(X) * ref_fraction))
    cut = min(cut, len(X) - 5)
    X_ref, X_cur = X[:cut], X[cut:]
    y_ref, y_cur = y[:cut], y[cut:]

    mlflow_info: dict[str, Any] = {"enabled": use_mlflow}
    run = None
    if use_mlflow:
        run = start_run(
            "gamma-squeeze-training",
            run_name=f"train-{'-'.join(syms[:3])}",
            params={
                "symbols": ",".join(syms),
                "n_trials": n_trials,
                "n_splits": n_splits,
                "validation_method": validation_method,
                "label_col": label_col,
            },
        )
        mlflow_info["run_active"] = run is not None

    # Bayesian hyperparameter search (+ early stopping inside)
    hyper = bayesian_search(
        X_ref,
        y_ref,
        n_trials=n_trials,
        n_splits=max(2, min(n_splits, 4)),
        early_stopping_rounds=8,
        patience_no_improve=max(5, n_trials // 2),
    )

    # Validation with best params (or defaults)
    params = hyper.best_params if hyper.status == "ok" else {}

    def fit_predict(Xtr, ytr, Xte):
        model = fit_best_classifier(Xtr, ytr, params)
        if model is None:
            return np.full(len(Xte), 0.5)
        proba = model.predict_proba(Xte)
        return proba[:, 1] if proba.ndim == 2 else proba

    method = "time_series_split" if validation_method == "time_series_split" else "walk_forward"
    val_report = evaluate_splits(X, y, fit_predict, method=method, n_splits=n_splits, expanding=True)

    # Fit reference model for drift scoring
    ref_model = fit_best_classifier(X_ref, y_ref, params)
    if ref_model is not None:
        scores_ref = ref_model.predict_proba(X_ref)[:, 1]
        scores_cur = ref_model.predict_proba(X_cur)[:, 1]
    else:
        scores_ref = np.full(len(X_ref), 0.5)
        scores_cur = np.full(len(X_cur), 0.5)

    drift = detect_all_drift(
        X_ref=X_ref,
        X_cur=X_cur,
        scores_ref=scores_ref,
        scores_cur=scores_cur,
        y_ref=y_ref,
        y_cur=y_cur,
        feature_names=cols,
    )
    decision = decide_retrain(
        drift,
        policy=AutoRetrainPolicy(
            force=force_retrain,
            on_feature_drift=auto_retrain,
            on_prediction_drift=auto_retrain,
            on_concept_drift=auto_retrain,
        ),
    )

    version_info: dict[str, Any] = {}
    metrics = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "validation_auc_mean": val_report.aggregate.get("auc_mean"),
        "hyperopt_best": hyper.best_value,
        "hyperopt_status": hyper.status,
        "early_stopped": bool(hyper.early_stopped),
        "drift_triggered": drift.triggered,
        "validation_method": method,
    }

    existing = list_versions("squeeze_probability")
    prior_auc = None
    for rec in reversed(existing):
        m = rec.get("metrics") or {}
        if m.get("validation_auc_mean") is not None:
            prior_auc = float(m["validation_auc_mean"])
            break

    candidate_auc = float(val_report.aggregate.get("auc_mean") or 0.5)
    # Drift (or empty registry / force) gates whether we fit a candidate;
    # promote only when AUC holds or improves vs active prior.
    should_fit = decision.should_retrain or force_retrain or not existing
    promote = False
    if should_fit:
        final_model = fit_best_classifier(X, y, params)
        if final_model is not None:
            if prior_auc is None or force_retrain or candidate_auc + 1e-4 >= prior_auc:
                promote = True
            ver = save_versioned_model(
                final_model,
                name="squeeze_probability",
                backend="training-pipeline",
                metrics={**metrics, "best_params": params, "prior_auc": prior_auc},
                tags={
                    "validation": method,
                    "early_stopping": str(hyper.early_stopped),
                    "decision": "promote" if promote else "hold",
                },
                activate=promote,
            )
            version_info = ver.to_dict()
            metrics["retrained"] = True
            metrics["registry_decision"] = "promote" if promote else "hold"
            metrics["prior_auc"] = prior_auc
            metrics["candidate_auc"] = candidate_auc
        else:
            metrics["retrained"] = False
            metrics["retrain_error"] = "fit_failed"
            metrics["registry_decision"] = "hold"
    else:
        metrics["retrained"] = False
        metrics["retrain_skipped"] = decision.reasons or ["no_drift"]
        metrics["registry_decision"] = "hold"

    if use_mlflow and run is not None:
        loggable = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        log_metrics(loggable)
        end_run()
        mlflow_info["logged"] = True

    return TrainingResult(
        symbols=syms,
        status="ok",
        validation=val_report.to_dict(),
        hyperopt=hyper.to_dict(),
        drift=drift.to_dict(),
        retrain_decision=decision.to_dict(),
        version=version_info,
        mlflow=mlflow_info,
        metrics=metrics,
    )


def training_meta() -> dict[str, Any]:
    from gamma_squeeze.stack.mlflow_tracking import health as mlflow_health

    try:
        import optuna  # noqa: F401

        optuna_ok = True
    except ImportError:
        optuna_ok = False

    return {
        "service": "model_training",
        "implement": list(TRAINING_CAPABILITIES),
        "validation_methods": ["walk_forward", "time_series_split"],
        "hyperopt": {
            "library": "optuna",
            "sampler": "TPESampler",
            "early_stopping": True,
            "available": optuna_ok,
        },
        "mlflow": mlflow_health(),
        "drift": ["Feature Drift", "Prediction Drift", "Concept Drift"],
        "drift_keys": ["feature", "prediction", "concept"],
        "version_control": "models/versions/{name}/{version}",
        "automatic_retraining": True,
        "endpoints": [
            "/v1/meta",
            "/v1/train",
            "/v1/drift",
            "/v1/versions",
            "/v1/rollback",
        ],
    }
