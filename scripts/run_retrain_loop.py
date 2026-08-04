#!/usr/bin/env python3
"""Continuous retraining entrypoint (baseline | rl | deep | training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.retrain.pipeline import run_retrain  # noqa: E402
from gamma_squeeze.training.pipeline import run_model_training  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True)
    p.add_argument("--mode", default="baseline", choices=["baseline", "rl", "deep", "training"])
    p.add_argument("--backend", default="", help="Alias for --mode when not using training")
    p.add_argument("--skip-features", action="store_true")
    p.add_argument("--skip-labels", action="store_true")
    p.add_argument("--no-infer", action="store_true")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument(
        "--validation",
        default="walk_forward",
        choices=["walk_forward", "time_series_split"],
    )
    p.add_argument("--force-retrain", action="store_true")
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    mode = args.backend or args.mode

    if mode == "training":
        result = run_model_training(
            symbols,
            rebuild=not args.skip_features,
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            validation_method=args.validation,
            use_mlflow=True,
            auto_retrain=True,
            force_retrain=args.force_retrain,
        )
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0 if result.status == "ok" else 1

    result = run_retrain(
        symbols,
        rebuild_features=not args.skip_features,
        rebuild_labels=not args.skip_labels,
        backend=mode,
        export_infer=not args.no_infer,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        validation_method=args.validation,
        use_mlflow=True,
        force_promote=args.force_retrain,
    )
    print(
        json.dumps(
            {
                "symbols": result.symbols,
                "metrics": result.metrics,
                "model_versions": result.model_versions,
                "forecast_paths": result.forecast_paths,
            },
            indent=2,
        )
    )
    return 0 if "error" not in result.metrics else 1


if __name__ == "__main__":
    raise SystemExit(main())
