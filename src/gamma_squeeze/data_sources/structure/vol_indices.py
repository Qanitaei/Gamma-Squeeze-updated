"""VIX / VVIX / MOVE adapters."""

from __future__ import annotations

import json
import ssl
import urllib.request
from typing import Any

from gamma_squeeze.data_sources.base import ok
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.ohlcv_client import fetch_daily_ohlcv

try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()


class VolIndicesAdapter:
    name = "structure.vol_indices"
    category = "structure"

    SYMBOLS = {
        "VIX": ["VIX", "^VIX", "VIXY"],
        "VVIX": ["VVIX", "^VVIX"],
        "MOVE": ["MOVE", "^MOVE"],
    }

    def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "configured": True, "indices": list(self.SYMBOLS)}

    def fetch(self) -> dict:
        out: dict[str, Any] = {}
        for label, candidates in self.SYMBOLS.items():
            val = None
            origin = None
            # CBOE delayed for VIX/VVIX
            if label in {"VIX", "VVIX"}:
                try:
                    url = f"https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{label}.json"
                    req = urllib.request.Request(url, headers={"User-Agent": "GammaSqueezePlatform/0.2"})
                    with urllib.request.urlopen(req, timeout=20, context=_SSL) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                    current = (payload.get("data") or {}).get("current_price") or (payload.get("data") or {}).get(
                        "last"
                    )
                    if current is not None:
                        val = float(current)
                        origin = "cboe_cdn"
                except Exception:  # noqa: BLE001
                    pass
            if val is None:
                for sym in candidates:
                    try:
                        df = fetch_daily_ohlcv(sym)
                        if not df.empty:
                            val = float(df["close"].iloc[-1])
                            origin = f"ohlcv:{sym}"
                            break
                    except Exception:  # noqa: BLE001
                        continue
            out[label] = {"value": val, "origin": origin}
        return ok(self.name, self.category, data={"indices": out}).to_dict()


register("structure.vol_indices", VolIndicesAdapter)
