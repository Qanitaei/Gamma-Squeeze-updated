#!/usr/bin/env python3
"""Upload local squeeze forecasts into Cloudflare KV (legacy alias).

Prefer ``upload_squeeze_forecasts_kv.py`` for gamma-squeeze-data with
per-extraction historical retention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from upload_squeeze_forecasts_kv import upload_extraction_batch  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stamp", default="", help="Extraction stamp; default=latest batch")
    p.add_argument("--symbols", default="", help="Comma-separated filter")
    args = p.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    summary = upload_extraction_batch(stamp=args.stamp.strip() or None, symbols=symbols)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
