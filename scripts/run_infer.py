#!/usr/bin/env python3
"""Run inference and export schema-complete 1–10d squeeze forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.config import resolve_matrix_root  # noqa: E402
from gamma_squeeze.features.feature_store import load_features, build_features_for_symbol  # noqa: E402
from gamma_squeeze.ingest.alpaca_backup_client import load_local_matrix  # noqa: E402
from gamma_squeeze.models.ensemble import SqueezeEnsemble  # noqa: E402
from gamma_squeeze.serve.export_forecasts import export_forecast  # noqa: E402
from gamma_squeeze.serve.schemas import validate_forecast_dict  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True)
    p.add_argument("--rebuild-features", action="store_true")
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: latest feature row)")
    args = p.parse_args()

    ensemble = SqueezeEnsemble.load_latest()
    mroot = resolve_matrix_root()
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        if args.rebuild_features:
            build_features_for_symbol(sym, include_skew=False, include_macro=False)
        feat = load_features(sym)
        if feat.empty:
            print(f"{sym}: no features — building from local matrices…")
            feat = build_features_for_symbol(sym, include_skew=False, include_macro=False)
        if feat.empty:
            print(f"{sym}: SKIP (no features)")
            continue
        if args.as_of:
            rows = feat[feat["as_of"].astype(str).str[:10] == args.as_of]
            if rows.empty:
                print(f"{sym}: no feature row for {args.as_of}")
                continue
            row = rows.iloc[-1]
        else:
            row = feat.sort_values("as_of").iloc[-1]
        as_of = str(row["as_of"])[:10]
        matrix = load_local_matrix(mroot, sym, as_of)
        forecast = ensemble.predict(row, symbol=sym, as_of=as_of, matrix=matrix, feature_panel=feat)
        path = export_forecast(forecast)
        errors = validate_forecast_dict(forecast.to_dict())
        print(f"{sym} {as_of} → {path} valid={not errors}")
        if errors:
            print("  errors:", errors[:5])
        else:
            h5 = next(h for h in forecast.horizons if h.horizon_days == 5)
            print(f"  P5={h5.squeeze_probability:.3f} mag={h5.expected_magnitude_pct:.2f}% conf={h5.confidence_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
