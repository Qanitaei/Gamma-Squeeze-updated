#!/usr/bin/env python3
"""Upload local squeeze forecasts into Cloudflare KV (SQUEEZE_FORECASTS)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.config import resolve_forecasts_root  # noqa: E402


def _put(account_id: str, namespace_id: str, token: str, key: str, value: str) -> None:
    enc = urllib.parse.quote(key, safe="")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{enc}"
    )
    req = urllib.request.Request(
        url,
        data=value.encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()


def main() -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default="", help="Comma-separated; default=all latest under forecasts/")
    args = p.parse_args()

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    namespace_id = os.getenv("CLOUDFLARE_SQUEEZE_NAMESPACE_ID", "").strip()
    if not (account_id and token and namespace_id):
        print("Set CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, CLOUDFLARE_SQUEEZE_NAMESPACE_ID")
        return 1

    root = resolve_forecasts_root()
    ticker_root = root / "by-ticker"
    if not ticker_root.is_dir():
        print(f"No forecasts at {ticker_root}")
        return 1

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = sorted(d.name for d in ticker_root.iterdir() if d.is_dir())

    index = {"symbols": symbols, "service": "gamma-squeeze-forecast"}
    uploaded = 0
    for sym in symbols:
        latest = ticker_root / sym / "latest.json"
        if not latest.is_file():
            print(f"{sym}: missing latest.json")
            continue
        payload = latest.read_text()
        data = json.loads(payload)
        as_of = str(data.get("as_of", ""))[:10]
        _put(account_id, namespace_id, token, f"{sym}/latest", payload)
        if as_of:
            _put(account_id, namespace_id, token, f"{sym}/forecast/{as_of}", payload)
        uploaded += 1
        print(f"uploaded {sym} as_of={as_of}")

    _put(account_id, namespace_id, token, "index", json.dumps(index))
    print(f"Done. uploaded={uploaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
