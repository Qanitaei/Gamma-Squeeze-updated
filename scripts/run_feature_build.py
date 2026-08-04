#!/usr/bin/env python3
"""Build feature store from local Gamma Squeeze Matrix (and optional live skew/macro)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.config import resolve_features_root, resolve_matrix_root  # noqa: E402
from gamma_squeeze.features.feature_store import (  # noqa: E402
    build_features_for_symbol,
    features_path,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True, help="Comma-separated tickers")
    p.add_argument("--no-skew", action="store_true")
    p.add_argument("--no-macro", action="store_true")
    p.add_argument("--no-ohlcv", action="store_true")
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    mroot = resolve_matrix_root()
    froot = resolve_features_root()
    print(f"Matrix root: {mroot}")
    print(f"Features root: {froot}")
    for sym in symbols:
        df = build_features_for_symbol(
            sym,
            matrix_root=mroot,
            features_root=froot,
            include_skew=not args.no_skew,
            include_macro=not args.no_macro,
            include_ohlcv=not args.no_ohlcv,
        )
        print(f"{sym}: {len(df)} feature rows → {features_path(sym, root=froot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
