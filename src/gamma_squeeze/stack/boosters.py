"""Unified tabular booster factory: XGBoost / LightGBM / CatBoost / sklearn fallback."""

from __future__ import annotations

from typing import Any, Literal

BoosterName = Literal["xgboost", "lightgbm", "catboost", "sklearn"]


def available_boosters() -> dict[str, bool]:
    out = {"xgboost": False, "lightgbm": False, "catboost": False, "sklearn": True}
    try:
        import xgboost  # noqa: F401

        out["xgboost"] = True
    except Exception:  # noqa: BLE001
        pass
    try:
        import lightgbm  # noqa: F401

        out["lightgbm"] = True
    except Exception:  # noqa: BLE001
        pass
    try:
        import catboost  # noqa: F401

        out["catboost"] = True
    except Exception:  # noqa: BLE001
        pass
    return out


def make_classifier(
    name: BoosterName = "xgboost",
    *,
    n_classes: int = 3,
    params: dict[str, Any] | None = None,
) -> Any:
    """Return a classifier instance; falls back toward sklearn if preferred lib missing."""
    p = {
        "n_estimators": 80,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "random_state": 42,
    }
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=int(p["n_estimators"]),
                max_depth=int(p["max_depth"]),
                learning_rate=float(p["learning_rate"]),
                subsample=float(p.get("subsample", 0.9)),
                colsample_bytree=float(p.get("colsample_bytree", 0.9)),
                min_child_weight=float(p.get("min_child_weight", 1.0)),
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=int(p["random_state"]),
                n_jobs=1,
            )
        except Exception:  # noqa: BLE001
            name = "lightgbm"
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=int(p["n_estimators"]),
                max_depth=int(p["max_depth"]),
                learning_rate=float(p["learning_rate"]),
                subsample=float(p.get("subsample", 0.9)),
                colsample_bytree=float(p.get("colsample_bytree", 0.9)),
                min_child_weight=float(p.get("min_child_weight", 1.0)),
                random_state=int(p["random_state"]),
                verbose=-1,
            )
        except Exception:  # noqa: BLE001
            name = "catboost"
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=int(p["n_estimators"]),
                depth=int(p["max_depth"]),
                learning_rate=float(p["learning_rate"]),
                verbose=False,
                random_seed=int(p["random_state"]),
                loss_function="MultiClass",
            )
        except Exception:  # noqa: BLE001
            pass
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_depth=int(p["max_depth"]),
        learning_rate=float(p["learning_rate"]),
        max_iter=int(p["n_estimators"]),
        random_state=int(p["random_state"]),
    )


def make_regressor(
    name: BoosterName = "xgboost",
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """Return a regressor instance with the same fallback chain as classifiers."""
    p = {
        "n_estimators": 80,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "random_state": 42,
    }
    if params:
        p.update({k: v for k, v in params.items() if v is not None})

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=int(p["n_estimators"]),
                max_depth=int(p["max_depth"]),
                learning_rate=float(p["learning_rate"]),
                subsample=float(p.get("subsample", 0.9)),
                colsample_bytree=float(p.get("colsample_bytree", 0.9)),
                min_child_weight=float(p.get("min_child_weight", 1.0)),
                objective="reg:squarederror",
                random_state=int(p["random_state"]),
                n_jobs=1,
            )
        except Exception:  # noqa: BLE001
            name = "lightgbm"
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=int(p["n_estimators"]),
                max_depth=int(p["max_depth"]),
                learning_rate=float(p["learning_rate"]),
                subsample=float(p.get("subsample", 0.9)),
                colsample_bytree=float(p.get("colsample_bytree", 0.9)),
                min_child_weight=float(p.get("min_child_weight", 1.0)),
                random_state=int(p["random_state"]),
                verbose=-1,
            )
        except Exception:  # noqa: BLE001
            name = "catboost"
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=int(p["n_estimators"]),
                depth=int(p["max_depth"]),
                learning_rate=float(p["learning_rate"]),
                verbose=False,
                random_seed=int(p["random_state"]),
                loss_function="RMSE",
            )
        except Exception:  # noqa: BLE001
            pass
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_depth=int(p["max_depth"]),
        learning_rate=float(p["learning_rate"]),
        max_iter=int(p["n_estimators"]),
        random_state=int(p["random_state"]),
    )
