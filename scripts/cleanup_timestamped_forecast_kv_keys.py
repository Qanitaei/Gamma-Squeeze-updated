#!/usr/bin/env python3
"""Delete KV keys that use T061106Z / T153655Z-style suffixes (keep YYYY-MM-DD only).

Removes:
  {TICKER}/forecast/YYYY-MM-DDTHHMMSSZ
  {TICKER}/pipeline/YYYYMMDDTHHMMSSZ
  scans/phase14_pipeline/YYYYMMDDTHHMMSSZ(+ /findings)
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
STAMPED = re.compile(
    r"^(?:"
    r"[A-Z.]{1,10}/forecast/\d{4}-\d{2}-\d{2}T\d{6}Z|"
    r"[A-Z.]{1,10}/pipeline/\d{8}T\d{6}Z|"
    r"scans/phase14_pipeline/\d{8}T\d{6}Z(?:/findings)?"
    r")$"
)


def _load_env() -> None:
    for path in (
        Path("/Users/ruslantkach/Desktop/schwab-options-export/.env"),
        ROOT / ".env",
    ):
        if path.is_file():
            load_dotenv(path, override=False)
    schwab = Path("/Users/ruslantkach/Desktop/schwab-options-export/.env")
    if schwab.is_file():
        for k, v in (dotenv_values(schwab) or {}).items():
            if v and not (os.getenv(k) or "").strip():
                os.environ[k] = v


def _credentials() -> tuple[str, str]:
    _load_env()
    cred = Path("/Users/ruslantkach/Desktop/economic-calendar/.cloudflare-credentials.json")
    if cred.is_file():
        data = json.loads(cred.read_text(encoding="utf-8"))
        return str(data["account_id"]).strip(), str(data["api_token"]).strip()
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token:
        raise SystemExit("Missing Cloudflare credentials")
    return account, token


def _namespace_id() -> str:
    meta = ROOT / "data" / "cloudflare_gamma_squeeze_data.json"
    if meta.is_file():
        return str(json.loads(meta.read_text())["namespace_id"])
    raise SystemExit("Missing namespace id")


def kv_list(account_id: str, token: str, namespace_id: str, cursor: str | None = None) -> dict:
    q = "limit=1000"
    if cursor:
        q += f"&cursor={urllib.parse.quote(cursor)}"
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/keys?{q}"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode())


def kv_delete(account_id: str, token: str, namespace_id: str, key: str) -> None:
    enc = urllib.parse.quote(key, safe="")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/storage/kv/namespaces/{namespace_id}/values/{enc}"
    )
    req = urllib.request.Request(url, method="DELETE", headers={"Authorization": f"Bearer {token}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                resp.read()
            return
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            if exc.code == 404:
                return
            raise


def main() -> int:
    account_id, token = _credentials()
    ns_id = _namespace_id()
    cursor = None
    stamped: list[str] = []
    while True:
        body = kv_list(account_id, token, ns_id, cursor)
        result = body.get("result") or []
        for row in result:
            name = row.get("name") if isinstance(row, dict) else str(row)
            if name and STAMPED.match(name):
                stamped.append(name)
        info = body.get("result_info") or {}
        if info.get("cursor") and not info.get("list_complete", True):
            cursor = info["cursor"]
            continue
        # CF sometimes only sets cursor
        if info.get("cursor") and len(result) >= 1000:
            cursor = info["cursor"]
            continue
        break

    print(json.dumps({"found_stamped_forecast_keys": len(stamped), "sample": stamped[:10]}, indent=2))
    for key in stamped:
        kv_delete(account_id, token, ns_id, key)
        print(f"deleted {key}")
    print(json.dumps({"deleted": len(stamped)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
