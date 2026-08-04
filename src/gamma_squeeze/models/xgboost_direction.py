"""XGBoost multi-horizon direction model — classification, regression, probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from gamma_squeeze.config import resolve_models_root
from gamma_squeeze.stack.boosters import available_boosters, make_classifier, make_regressor
from gamma_squeeze.stack.optuna_search import run_study
from gamma_squeeze.stack.shap_explain import shap_summary

DIRECTION_LABELS = {0: "down", 1: "flat", 2: "up"}
DIRECTION_HORIZONS = (1, 3, 5, 10, 20)
DIRECTION_HORIZON_LABELS = {
    1: "1 Day Return",
    3: "3 Day Return",
    5: "5 Day Return",
    10: "10 Day Return",
    20: "20 Day Return",
}
DIRECTION_TARGETS = ("classification", "regression", "probability")
DIRECTION_TRAINING = (
    "cross_validation",  # TimeSeriesSplit
    "bayesian_optimization",  # Optuna TPE
    "shap_explainability",
)
_AVAIL = available_boosters()
HAS_XGB = bool(_AVAIL.get("xgboost"))
DEFAULT_BACKEND = (
    "xgboost"
    if HAS_XGB
    else ("lightgbm" if _AVAIL.get("lightgbm") else ("catboost" if _AVAIL.get("catboost") else "sklearn"))
)


@dataclass
class HorizonBundle:
    horizon: int
    classifier: Any = None
    regressor: Any = None
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_scores: dict[str, Any] = field(default_factory=dict)
    shap: dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectionModelBundle:
    models: dict[int, HorizonBundle] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    horizons: list[int] = field(default_factory=lambda: list(DIRECTION_HORIZONS))
    direction_threshold: float = 0.005
    backend: str = DEFAULT_BACKEND
    version: str = "xgb-direction-multih-v2"
    # legacy single-model fields (older checkpoints)
    model: Any = None

    def predict_row(
        self,
        row: dict[str, Any] | pd.Series,
        *,
        horizon: int | None = None,
    ) -> dict[str, Any]:
        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        vals: list[float] = []
        for c in self.feature_columns:
            try:
                v = float(data.get(c))
            except (TypeError, ValueError):
                v = 0.0
            if not np.isfinite(v):
                v = 0.0
            vals.append(v)
        x = np.asarray([vals], dtype=float)

        # Legacy single classifier checkpoint
        if not self.models and self.model is not None:
            return _predict_classifier_only(self.model, x, horizon=horizon or 5, version=self.version)

        if horizon is not None:
            hb = self.models.get(int(horizon))
            if hb is None:
                return {
                    "direction": "flat",
                    "classification": "flat",
                    "regression": 0.0,
                    "probability": {"down": 0.33, "flat": 0.34, "up": 0.33},
                    "horizon": horizon,
                }
            return _predict_horizon(hb, x, version=self.version)

        horizons_out: dict[str, Any] = {}
        for h in self.horizons:
            hb = self.models.get(int(h))
            if hb is None:
                continue
            horizons_out[f"{h}d"] = _predict_horizon(hb, x, version=self.version)

        # Convenience: expose primary 5d (or first available) at top level
        primary_key = "5d" if "5d" in horizons_out else (next(iter(horizons_out), None))
        primary = horizons_out.get(primary_key, {}) if primary_key else {}
        return {
            "horizons": horizons_out,
            "direction": primary.get("classification", "flat"),
            "direction_id": primary.get("direction_id", 1),
            "proba": primary.get("probability", {}),
            "classification": primary.get("classification"),
            "regression": primary.get("regression"),
            "probability": primary.get("probability"),
            "version": self.version,
        }


def _predict_classifier_only(model: Any, x: np.ndarray, *, horizon: int, version: str) -> dict[str, Any]:
    pred = int(model.predict(x)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        proba_map = {DIRECTION_LABELS.get(i, str(i)): float(p) for i, p in enumerate(proba)}
    else:
        proba_map = {DIRECTION_LABELS.get(pred, "flat"): 1.0}
    return {
        "horizon": horizon,
        "classification": DIRECTION_LABELS.get(pred, "flat"),
        "direction": DIRECTION_LABELS.get(pred, "flat"),
        "direction_id": pred,
        "regression": None,
        "probability": proba_map,
        "proba": proba_map,
        "version": version,
    }


def _predict_horizon(hb: HorizonBundle, x: np.ndarray, *, version: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "horizon": hb.horizon,
        "horizon_label": DIRECTION_HORIZON_LABELS.get(hb.horizon, f"{hb.horizon} Day Return"),
        "classification": "flat",
        "direction": "flat",
        "direction_id": 1,
        "regression": 0.0,  # predicted forward return
        "probability": {"down": 0.33, "flat": 0.34, "up": 0.33},
        "confidence": 0.0,
        "targets": list(DIRECTION_TARGETS),
        "version": version,
    }
    if hb.classifier is not None:
        pred = int(hb.classifier.predict(x)[0])
        out["classification"] = DIRECTION_LABELS.get(pred, "flat")
        out["direction"] = out["classification"]
        out["direction_id"] = pred
        if hasattr(hb.classifier, "predict_proba"):
            proba = hb.classifier.predict_proba(x)[0]
            # align to 3-class map when binary edge cases occur
            proba_map = {"down": 0.0, "flat": 0.0, "up": 0.0}
            classes = list(getattr(hb.classifier, "classes_", range(len(proba))))
            for i, p in enumerate(proba):
                label = DIRECTION_LABELS.get(int(classes[i]), str(classes[i]))
                if label in proba_map:
                    proba_map[label] = float(p)
            s = sum(proba_map.values()) or 1.0
            proba_map = {k: v / s for k, v in proba_map.items()}
            out["probability"] = proba_map
            out["proba"] = proba_map
            out["confidence"] = float(max(proba_map.values()))
    if hb.regressor is not None:
        out["regression"] = float(hb.regressor.predict(x)[0])
    return out


def _label_direction(fwd: float, thr: float = 0.005) -> int:
    if not np.isfinite(fwd):
        return 1
    if fwd > thr:
        return 2
    if fwd < -thr:
        return 0
    return 1


def _finite_forward_from_spot(spot: pd.Series, horizon: int) -> pd.Series:
    """Synthesize forward returns; zero/non-finite spots → NaN (never ±inf)."""
    s = pd.to_numeric(spot, errors="coerce")
    future = s.shift(-horizon)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = future / s - 1.0
    out = out.where((s > 0) & np.isfinite(s) & np.isfinite(future) & np.isfinite(out))
    return out


def _sanitize_target(series: pd.Series) -> pd.Series:
    """Coerce to float and drop non-finite (NaN/±inf) values to NaN."""
    y = pd.to_numeric(series, errors="coerce")
    arr = np.asarray(y, dtype=float)
    return y.where(np.isfinite(arr))


def _ensure_forward_returns(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Merge features/labels and synthesize missing fwd_ret_{h}d from spot."""
    feat = features.copy()
    if "as_of" in feat.columns:
        # preserve panel order within symbol when multi-ticker
        if "symbol" in feat.columns:
            feat = feat.sort_values(["symbol", "as_of"])
        else:
            feat = feat.sort_values("as_of")
    if labels is not None and not labels.empty:
        lab_cols = ["symbol", "as_of"] + [c for c in labels.columns if c.startswith("fwd_ret_")]
        lab_cols = [c for c in lab_cols if c in labels.columns]
        merged = feat.merge(labels[lab_cols], on=["symbol", "as_of"], how="left", suffixes=("", "_lab"))
    else:
        merged = feat

    if "spot" in merged.columns:
        # synthesize per-symbol so shift(-h) does not cross ticker boundaries
        if "symbol" in merged.columns:
            synth_by_h = {
                h: merged.groupby("symbol", group_keys=False)["spot"].transform(
                    lambda s, hh=h: _finite_forward_from_spot(s, hh)
                )
                for h in horizons
            }
        else:
            synth_by_h = {h: _finite_forward_from_spot(merged["spot"], h) for h in horizons}
        for h in horizons:
            col = f"fwd_ret_{h}d"
            synth = synth_by_h[h]
            if col not in merged.columns or _sanitize_target(merged[col]).isna().mean() > 0.5:
                merged[col] = synth
            else:
                cleaned = _sanitize_target(merged[col])
                merged[col] = cleaned.where(cleaned.notna(), synth)
    for h in horizons:
        col = f"fwd_ret_{h}d"
        if col in merged.columns:
            merged[col] = _sanitize_target(merged[col])
    return merged


def _cv_split(n: int, n_splits: int) -> TimeSeriesSplit:
    splits = max(2, min(n_splits, max(2, n // 15)))
    return TimeSeriesSplit(n_splits=splits)


def _suggest_params(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 40, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
    }


def _optimize_horizon(
    X: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    *,
    backend: str,
    n_classes: int,
    n_trials: int,
    cv_folds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bayesian optimization of shared tree params via CV log-loss + RMSE."""
    tscv = _cv_split(len(X), cv_folds)

    def objective(trial: Any) -> float:
        params = _suggest_params(trial)
        cls_losses: list[float] = []
        reg_rmses: list[float] = []
        for train_idx, test_idx in tscv.split(X):
            if len(np.unique(y_cls[train_idx])) < 2:
                continue
            clf = make_classifier(backend, n_classes=n_classes, params=params)  # type: ignore[arg-type]
            reg = make_regressor(backend, params=params)  # type: ignore[arg-type]
            clf.fit(X[train_idx], y_cls[train_idx])
            reg.fit(X[train_idx], y_reg[train_idx])
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X[test_idx])
                # pad/align classes for log_loss
                labels = list(range(n_classes))
                try:
                    cls_losses.append(float(log_loss(y_cls[test_idx], proba, labels=labels)))
                except Exception:  # noqa: BLE001
                    pred = clf.predict(X[test_idx])
                    cls_losses.append(1.0 - float(accuracy_score(y_cls[test_idx], pred)))
            else:
                pred = clf.predict(X[test_idx])
                cls_losses.append(1.0 - float(accuracy_score(y_cls[test_idx], pred)))
            pred_r = reg.predict(X[test_idx])
            reg_rmses.append(float(np.sqrt(mean_squared_error(y_reg[test_idx], pred_r))))
        if not cls_losses:
            return 1e6
        # minimize blended loss
        return float(np.mean(cls_losses) + 0.5 * np.mean(reg_rmses))

    if n_trials <= 0:
        return {
            "n_estimators": 80,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
        }, {"status": "skipped", "n_trials": 0}

    study = run_study(objective, n_trials=n_trials, study_name="xgb-direction", direction="minimize")
    params = study.get("best_params") or {}
    if not params:
        params = {
            "n_estimators": 80,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
        }
    return params, study


def _evaluate_cv(
    X: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    *,
    backend: str,
    params: dict[str, Any],
    n_classes: int,
    cv_folds: int,
) -> dict[str, Any]:
    tscv = _cv_split(len(X), cv_folds)
    accs: list[float] = []
    rmses: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        if len(np.unique(y_cls[train_idx])) < 2:
            continue
        clf = make_classifier(backend, n_classes=n_classes, params=params)  # type: ignore[arg-type]
        reg = make_regressor(backend, params=params)  # type: ignore[arg-type]
        clf.fit(X[train_idx], y_cls[train_idx])
        reg.fit(X[train_idx], y_reg[train_idx])
        accs.append(float(accuracy_score(y_cls[test_idx], clf.predict(X[test_idx]))))
        rmses.append(float(np.sqrt(mean_squared_error(y_reg[test_idx], reg.predict(X[test_idx])))))
    return {
        "cv_folds": int(getattr(tscv, "n_splits", cv_folds)),
        "classification_accuracy_mean": float(np.mean(accs)) if accs else None,
        "classification_accuracy_std": float(np.std(accs)) if accs else None,
        "regression_rmse_mean": float(np.mean(rmses)) if rmses else None,
        "regression_rmse_std": float(np.std(rmses)) if rmses else None,
        "n_folds_scored": len(accs),
    }


def train_direction_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_columns: list[str],
    *,
    horizon: int | None = None,
    horizons: list[int] | tuple[int, ...] | None = None,
    direction_threshold: float = 0.005,
    n_trials: int = 15,
    cv_folds: int = 5,
    include_shap: bool = True,
    backend: str | None = None,
) -> DirectionModelBundle:
    """
    Train multi-horizon direction models.

    For each horizon (default 1/3/5/10/20d):
      - classification (down / flat / up)
      - regression (expected forward return)
      - class probabilities
    Training uses time-series CV + Optuna Bayesian optimization + SHAP.
    """
    hs = list(horizons) if horizons is not None else list(DIRECTION_HORIZONS)
    if horizon is not None and horizon not in hs:
        hs = sorted(set(hs) | {int(horizon)})
    backend_name = backend or DEFAULT_BACKEND

    bundle = DirectionModelBundle(
        feature_columns=list(feature_columns),
        horizons=hs,
        direction_threshold=direction_threshold,
        backend=backend_name,
        version=f"{backend_name}-direction-multih-v2",
    )

    merged = _ensure_forward_returns(features, labels, hs)
    if merged.empty or not feature_columns:
        return bundle

    X_all = merged[feature_columns].apply(pd.to_numeric, errors="coerce")
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for h in hs:
        y_col = f"fwd_ret_{h}d"
        if y_col not in merged.columns:
            continue
        y_clean = _sanitize_target(merged[y_col])
        mask = y_clean.notna()
        if int(mask.sum()) < 25:
            continue
        y_reg = y_clean.loc[mask].astype(float).to_numpy()
        # final guard — sklearn/xgboost reject non-finite targets
        finite = np.isfinite(y_reg)
        if int(finite.sum()) < 25:
            continue
        y_reg = y_reg[finite]
        y_cls = np.asarray([_label_direction(float(v), direction_threshold) for v in y_reg], dtype=int)
        if len(np.unique(y_cls)) < 2:
            continue
        X = np.asarray(X_all.loc[mask].to_numpy(), dtype=float)[finite]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n_classes = 3

        params, study = _optimize_horizon(
            X,
            y_cls,
            y_reg,
            backend=backend_name,
            n_classes=n_classes,
            n_trials=n_trials,
            cv_folds=cv_folds,
        )
        cv_scores = _evaluate_cv(
            X,
            y_cls,
            y_reg,
            backend=backend_name,
            params=params,
            n_classes=n_classes,
            cv_folds=cv_folds,
        )
        cv_scores["optuna"] = {
            "status": study.get("status"),
            "best_value": study.get("best_value"),
            "n_trials": study.get("n_trials"),
        }

        clf = make_classifier(backend_name, n_classes=n_classes, params=params)  # type: ignore[arg-type]
        reg = make_regressor(backend_name, params=params)  # type: ignore[arg-type]
        clf.fit(X, y_cls)
        reg.fit(X, y_reg)

        shap_cls: dict[str, Any] = {"status": "skipped"}
        shap_reg: dict[str, Any] = {"status": "skipped"}
        if include_shap:
            shap_cls = shap_summary(clf, X, feature_columns, max_samples=min(80, len(X)))
            shap_reg = shap_summary(reg, X, feature_columns, max_samples=min(80, len(X)))

        bundle.models[int(h)] = HorizonBundle(
            horizon=int(h),
            classifier=clf,
            regressor=reg,
            best_params=params,
            cv_scores=cv_scores,
            shap={"classification": shap_cls, "regression": shap_reg},
        )

    # legacy alias: 5d classifier if present
    if 5 in bundle.models:
        bundle.model = bundle.models[5].classifier
    elif bundle.models:
        bundle.model = next(iter(bundle.models.values())).classifier
    return bundle


def save_direction_model(bundle: DirectionModelBundle, *, root: Path | None = None) -> Path:
    path = (root or resolve_models_root()) / "xgboost_direction" / "latest.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_direction_model(*, root: Path | None = None) -> DirectionModelBundle | None:
    path = (root or resolve_models_root()) / "xgboost_direction" / "latest.joblib"
    if not path.is_file():
        return None
    return joblib.load(path)
