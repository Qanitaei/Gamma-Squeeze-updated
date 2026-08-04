#!/usr/bin/env python3
"""Print deploy hints for cloudflare-worker-squeeze (wrangler)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "cloudflare-worker-squeeze"


def main() -> int:
    print("Gamma Squeeze Forecast worker deploy:")
    print(f"  cd {WORKER}")
    print("  # 1) Create KV namespace 'gamma-squeeze-forecasts' in Cloudflare dashboard")
    print("  # 2) Set [[kv_namespaces]].id in wrangler.toml")
    print("  # 3) npm install && npx wrangler deploy")
    print("  # 4) Set CLOUDFLARE_SQUEEZE_NAMESPACE_ID + upload via scripts/upload_kv_squeeze_forecasts.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
