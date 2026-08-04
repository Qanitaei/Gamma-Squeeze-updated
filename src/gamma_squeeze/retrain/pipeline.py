"""End-to-end retrain: features → labels → train → validate → register → infer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gamma_squeeze.config import HORIZONS, resolve_features_root, resolve_models_root
from gamma_squeeze.features.feature_store import (
    FEATURE_COLUMNS,
    build_features_for_symbol,
    load_features,
    model_feature_matrix,
)
from gamma_squeeze.labels.label_builder import build_labels_for_symbol, load_labels
from gamma_squeeze.models.ensemble import SqueezeEnsemble, train_ensemble_components
from gamma_squeeze.retrain.registry import list_backends, register_model
from gamma_squeeze.serve.export_forecasts import export_forecast


@dataclass
class RetrainResult:
    symbols: list[str]
    metrics: dict[str, Any]
    model_versions: dict[str, str]
    forecast_paths: list[str]


def walk_forward_auc_proxy(labels: pd.DataFrame, probs: pd.Series, y_col: str) -> float:
    """Simple ranking metric when sklearn.metrics may be heavy — Mann-Whitney style."""
    y = labels[y_col].fillna(0).astype(int)
    pos = probs[y == 1]
    neg = probs[y == 0]
    if pos.empty or neg.empty:
        return 0.5
    # P(score_pos > score_neg)
    wins = 0.0
    for p in pos:
        wins += float((neg < p).mean())
    return float(wins / len(pos))


def run_retrain(
    symbols: list[str],
    *,
    rebuild_features: bool = True,
    rebuild_labels: bool = True,
    backend: str = "baseline",
    export_infer: bool = True,
    n_trials: int = 15,
    n_splits: int = 4,
    validation_method: str = "walk_forward",
    use_mlflow: bool = True,
    force_promote: bool = False,
) -> RetrainResult:
    """Train baseline ensemble (or invoke registered backend) on symbol list.

    Hardened ops path:
      - walk-forward / time-series validation
      - Optuna search when installed (skipped gracefully otherwise)
      - early stopping inside hyperopt
      - MLflow tracking when available
      - model registry under SSD models/
      - drift gating promote-or-hold
    """
    # Ensure stub backends are imported
    from gamma_squeeze.rl import agent as _rl  # noqa: F401
    from gamma_squeeze.deep import temporal_model as _dl  # noqa: F401

    feat_frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    for sym in symbols:
        if rebuild_features:
            f = build_features_for_symbol(sym, include_skew=False, include_macro=False)
        else:
            f = load_features(sym)
        if rebuild_labels:
            lab = build_labels_for_symbol(sym, features=f)
        else:
            lab = load_labels(sym)
        if not f.empty:
            feat_frames.append(f)
        if not lab.empty:
            label_frames.append(lab)

    if not feat_frames or not label_frames:
        return RetrainResult(symbols=symbols, metrics={"error": "no_data"}, model_versions={}, forecast_paths=[])

    features = pd.concat(feat_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)
    X, cols = model_feature_matrix(features)
    # keep only columns present
    cols = [c for c in cols if c in FEATURE_COLUMNS or c in features.columns]

    metrics: dict[str, Any] = {
        "backends_available": list_backends(),
        "backend_used": backend,
        "validation_method": validation_method,
        "models_root": str(resolve_models_root()),
        "features_root": str(resolve_features_root()),
    }
    model_versions: dict[str, str] = {}
    ensemble: SqueezeEnsemble

    # Optional MLflow
    mlflow_run = None
    if use_mlflow:
        try:
            from gamma_squeeze.stack.mlflow_tracking import end_run, log_metrics, start_run

            mlflow_run = start_run(
                "gamma-squeeze-retrain",
                run_name=f"retrain-{backend}-{'-'.join(s.upper() for s in symbols[:3])}",
                params={
                    "backend": backend,
                    "n_trials": n_trials,
                    "n_splits": n_splits,
                    "validation_method": validation_method,
                    "symbols": ",".join(symbols),
                },
            )
            metrics["mlflow"] = {"enabled": True, "run_active": mlflow_run is not None}
        except Exception as exc:  # noqa: BLE001
            metrics["mlflow"] = {"enabled": False, "error": str(exc)}
    else:
        metrics["mlflow"] = {"enabled": False}

    if backend == "baseline":
        # Hyperopt (Optuna) — skip gracefully if unavailable
        from gamma_squeeze.training.hyperopt import bayesian_search, fit_best_classifier
        from gamma_squeeze.training.validation import align_xy, evaluate_splits
        from gamma_squeeze.training.drift import detect_all_drift
        from gamma_squeeze.training.auto_retrain import AutoRetrainPolicy, decide_retrain

        X_arr, y_arr, merged = align_xy(features, labels, cols, label_col="squeeze_5d")
        hyper_status = "skipped"
        best_params: dict[str, Any] = {}
        if len(X_arr) >= 30 and len(np.unique(y_arr)) >= 2:
            hyper = bayesian_search(
                X_arr,
                y_arr,
                n_trials=n_trials,
                n_splits=max(2, min(n_splits, 4)),
                early_stopping_rounds=8,
                patience_no_improve=max(5, n_trials // 2),
            )
            hyper_status = hyper.status
            best_params = hyper.best_params if hyper.status == "ok" else {}
            metrics["hyperopt"] = hyper.to_dict()
        else:
            metrics["hyperopt"] = {"status": "insufficient_data"}

        method = "time_series_split" if validation_method == "time_series_split" else "walk_forward"
        if len(X_arr) >= 20 and len(np.unique(y_arr)) >= 2:

            def fit_predict(Xtr, ytr, Xte):
                model = fit_best_classifier(Xtr, ytr, best_params)
                if model is None:
                    return np.full(len(Xte), 0.5)
                proba = model.predict_proba(Xte)
                return proba[:, 1] if proba.ndim == 2 else proba

            val_report = evaluate_splits(
                X_arr, y_arr, fit_predict, method=method, n_splits=n_splits, expanding=True
            )
            metrics["validation"] = val_report.to_dict()
            candidate_auc = float(val_report.aggregate.get("auc_mean") or 0.5)
        else:
            candidate_auc = 0.5
            metrics["validation"] = {"status": "insufficient_data", "aggregate": {"auc_mean": 0.5}}

        # Drift on temporal halves
        if len(X_arr) >= 30:
            cut = max(10, int(len(X_arr) * 0.6))
            cut = min(cut, len(X_arr) - 5)
            ref_model = fit_best_classifier(X_arr[:cut], y_arr[:cut], best_params)
            if ref_model is not None:
                scores_ref = ref_model.predict_proba(X_arr[:cut])[:, 1]
                scores_cur = ref_model.predict_proba(X_arr[cut:])[:, 1]
            else:
                scores_ref = np.full(cut, 0.5)
                scores_cur = np.full(len(X_arr) - cut, 0.5)
            drift = detect_all_drift(
                X_ref=X_arr[:cut],
                X_cur=X_arr[cut:],
                scores_ref=scores_ref,
                scores_cur=scores_cur,
                y_ref=y_arr[:cut],
                y_cur=y_arr[cut:],
                feature_names=cols,
            )
            decision = decide_retrain(drift, policy=AutoRetrainPolicy(force=force_promote))
            metrics["drift"] = drift.to_dict()
            metrics["retrain_decision"] = decision.to_dict()
        else:
            decision = decide_retrain(
                {"feature_drift": {}, "prediction_drift": {}, "concept_drift": {}, "triggered": False, "reasons": []},
                policy=AutoRetrainPolicy(force=True),
            )
            metrics["drift"] = {"status": "skipped"}
            metrics["retrain_decision"] = decision.to_dict()

        # Always fit ensemble components for infer path; registry promote gated
        ensemble = train_ensemble_components(features, labels, cols)
        model_versions = {
            "probability": ensemble.probability.version,
            "magnitude": ensemble.magnitude.version,
            "duration": ensemble.duration.version,
        }
        if not merged.empty and ensemble.probability.models.get(5):
            probs = merged.apply(lambda r: ensemble.probability.predict_proba_row(r).get(5, 0.0), axis=1)
            metrics["auc_proxy_5d"] = walk_forward_auc_proxy(merged, probs, "squeeze_5d")
            metrics["positive_rate_5d"] = float(merged["squeeze_5d"].mean()) if "squeeze_5d" in merged else None

        from gamma_squeeze.retrain.registry import load_registry

        reg = load_registry()
        prior_auc = None
        for m in reversed(reg.get("models") or []):
            if m.get("name") == "squeeze_ensemble":
                prior_auc = (m.get("metrics") or {}).get("validation_auc_mean") or (m.get("metrics") or {}).get(
                    "auc_proxy_5d"
                )
                if prior_auc is not None:
                    prior_auc = float(prior_auc)
                    break

        has_active = bool((reg.get("active") or {}).get("squeeze_ensemble"))
        auc_ok = prior_auc is None or candidate_auc + 1e-4 >= float(prior_auc)
        if force_promote or not has_active:
            promote = True
        elif decision.should_retrain and auc_ok:
            promote = True
        else:
            promote = False

        metrics["validation_auc_mean"] = candidate_auc
        metrics["prior_auc"] = prior_auc
        metrics["registry_decision"] = "promote" if promote else "hold"
        metrics["hyperopt_status"] = hyper_status

        register_model(
            name="squeeze_ensemble",
            backend="baseline",
            version=ensemble.probability.version,
            path=resolve_models_root() / "probability" / "latest.joblib",
            metrics=metrics,
            activate=promote,
        )
    else:
        factory = None
        from gamma_squeeze.retrain.registry import get_backend

        factory = get_backend(backend)
        if factory is None:
            metrics["error"] = f"unknown_backend:{backend}"
            return RetrainResult(symbols, metrics, {}, [])
        result = factory(features=features, labels=labels, feature_columns=cols)
        model_versions = result.get("versions") or {backend: "stub"}
        metrics.update(result.get("metrics") or {})
        register_model(
            name="squeeze_ensemble",
            backend=backend,
            version=str(model_versions.get(backend, "stub")),
            path=resolve_models_root() / backend,
            metrics=metrics,
            activate=True,
        )
        ensemble = SqueezeEnsemble.load_latest()

    forecast_paths: list[str] = []
    if export_infer:
        for sym in symbols:
            f = load_features(sym)
            if f.empty:
                continue
            latest = f.sort_values("as_of").iloc[-1]
            as_of = str(latest["as_of"])[:10]
            from gamma_squeeze.ingest.alpaca_backup_client import load_local_matrix
            from gamma_squeeze.config import resolve_matrix_root

            matrix = load_local_matrix(resolve_matrix_root(), sym, as_of)
            forecast = ensemble.predict(latest, symbol=sym, as_of=as_of, matrix=matrix, feature_panel=f)
            path = export_forecast(forecast)
            forecast_paths.append(str(path))

    metrics["n_feature_rows"] = int(len(features))
    metrics["n_label_rows"] = int(len(labels))
    metrics["horizons"] = list(HORIZONS)

    if use_mlflow and mlflow_run is not None:
        try:
            from gamma_squeeze.stack.mlflow_tracking import end_run, log_metrics

            loggable = {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if loggable:
                log_metrics(loggable)
            end_run()
        except Exception:  # noqa: BLE001
            pass

    resolve_features_root()
    return RetrainResult(symbols=list(symbols), metrics=metrics, model_versions=model_versions, forecast_paths=forecast_paths)
