#!/usr/bin/env python3
"""Delete KV keys that use T061106Z / T153655Z-style suffixes (keep YYYY-MM-DD only).

Removes:
  {TICKER}/forecast/YYYY-MM-DDTHHMMSSZ
  {TICKER}/pipeline/YYYYMMDDTHHMMSSZ
  scans/phase14_pipeline/YYYYMMDDTHHMMSSZ(+ /findings)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gamma_squeeze.cloudflare_kv import (  # noqa: E402
    cloudflare_credentials,
    gamma_squeeze_namespace_id,
    kv_delete,
    kv_list_keys,
)

STAMPED = re.compile(
    r"^(?:"
    r"[A-Z.]{1,10}/forecast/\d{4}-\d{2}-\d{2}T\d{6}Z|"
    r"[A-Z.]{1,10}/pipeline/\d{8}T\d{6}Z|"
    r"scans/phase14_pipeline/\d{8}T\d{6}Z(?:/findings)?"
    r")$"
)


def main() -> int:
    ns_id = gamma_squeeze_namespace_id()
    account_id, token = cloudflare_credentials(namespace_id=ns_id)
    keys = kv_list_keys(account_id, token, ns_id)
    stamped = [name for name in keys if STAMPED.match(name)]
    print(
        json.dumps(
            {"found_stamped_forecast_keys": len(stamped), "sample": stamped[:10]},
            indent=2,
        )
    )
    for key in stamped:
        kv_delete(account_id, token, ns_id, key)
        print(f"deleted {key}")
    print(json.dumps({"deleted": len(stamped)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
