#!/usr/bin/env python3
"""CLI: top-N market-cap × 14-stage pipeline → PortableSSD export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--limit", type=int, default=None, help="Optional cap for smoke runs")
    p.add_argument("--refresh-universe", action="store_true", default=True)
    p.add_argument("--no-refresh-universe", action="store_true")
    p.add_argument("--train-models", action="store_true")
    p.add_argument("--rebuild-features", action="store_true", default=True)
    p.add_argument("--no-rebuild-features", action="store_true")
    p.add_argument("--persist-full", action="store_true")
    p.add_argument("--sync-matrices", action="store_true", default=True)
    p.add_argument("--no-sync-matrices", action="store_true")
    p.add_argument(
        "--matrix-lookback-dates",
        type=int,
        default=10,
        help="Trailing option-matrix dates to sync from KV (default 10)",
    )
    args = p.parse_args()

    from gamma_squeeze.scan.universe_pipeline import run_top100_pipeline

    payload = run_top100_pipeline(
        top=args.top,
        refresh_universe=not args.no_refresh_universe,
        rebuild_features=args.rebuild_features and not args.no_rebuild_features,
        train_models=args.train_models,
        limit=args.limit,
        persist_full_results=args.persist_full,
        sync_matrices=args.sync_matrices and not args.no_sync_matrices,
        matrix_lookback_dates=args.matrix_lookback_dates,
    )
    summary = {
        "export_dir": payload.get("export_dir"),
        "scan_id": payload.get("scan_id"),
        "n_scanned": payload.get("n_scanned"),
        "n_pipeline_ok": payload.get("n_pipeline_ok"),
        "matrix_sync": payload.get("matrix_sync"),
        "top_squeeze": [
            {
                "symbol": r.get("symbol"),
                "p": r.get("squeeze_probability"),
                "regime": r.get("regime"),
                "ppo": r.get("ppo_action"),
                "alerts": r.get("n_alerts"),
            }
            for r in (payload.get("top_squeeze_candidates") or [])[:15]
        ],
        "stage_fail_counts": payload.get("stage_fail_counts"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
