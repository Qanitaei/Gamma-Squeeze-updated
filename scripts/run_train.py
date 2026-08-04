#!/usr/bin/env python3
"""Train baseline probability / magnitude / duration models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.features.feature_store import load_features, model_feature_matrix  # noqa: E402
from gamma_squeeze.labels.label_builder import load_labels  # noqa: E402
from gamma_squeeze.models.ensemble import train_ensemble_components  # noqa: E402
from gamma_squeeze.retrain.registry import register_model  # noqa: E402
from gamma_squeeze.config import resolve_models_root  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True)
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    feats = []
    labs = []
    for sym in symbols:
        f = load_features(sym)
        l = load_labels(sym)
        if not f.empty:
            feats.append(f)
        if not l.empty:
            labs.append(l)
    if not feats or not labs:
        print("No features/labels found. Run feature + label build first (and Phase 2 SSD export).")
        return 1
    features = pd.concat(feats, ignore_index=True)
    labels = pd.concat(labs, ignore_index=True)
    _, cols = model_feature_matrix(features)
    ens = train_ensemble_components(features, labels, cols)
    register_model(
        name="squeeze_ensemble",
        backend="baseline",
        version=ens.probability.version,
        path=resolve_models_root() / "probability" / "latest.joblib",
        metrics={"n_rows": len(features)},
        activate=True,
    )
    print(f"Trained: {ens.probability.version} / {ens.magnitude.version} / {ens.duration.version}")
    print(f"Models root: {resolve_models_root()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
