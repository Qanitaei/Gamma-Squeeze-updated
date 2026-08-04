#!/usr/bin/env python3
"""Build multi-horizon squeeze labels from features + OHLCV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.config import resolve_features_root, resolve_matrix_root  # noqa: E402
from gamma_squeeze.labels.label_builder import build_labels_for_symbol, labels_path  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True)
    args = p.parse_args()
    froot = resolve_features_root()
    print(f"Matrix root: {resolve_matrix_root()}")
    print(f"Features/labels root: {froot}")
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        df = build_labels_for_symbol(sym, features_root=froot)
        pos = int(df["squeeze_5d"].sum()) if not df.empty and "squeeze_5d" in df.columns else 0
        print(f"{sym}: {len(df)} label rows, squeeze_5d positives={pos} → {labels_path(sym, root=froot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
