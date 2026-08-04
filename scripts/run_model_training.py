#!/usr/bin/env python3
"""Run walk-forward / Bayesian training with MLflow + drift auto-retrain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.training.pipeline import run_model_training  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument(
        "--validation",
        default="walk_forward",
        choices=["walk_forward", "time_series_split"],
    )
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-auto-retrain", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    result = run_model_training(
        symbols,
        rebuild=args.rebuild,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        validation_method=args.validation,
        use_mlflow=not args.no_mlflow,
        auto_retrain=not args.no_auto_retrain,
        force_retrain=args.force_retrain,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
