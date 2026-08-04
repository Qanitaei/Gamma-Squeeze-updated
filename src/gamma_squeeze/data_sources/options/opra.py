"""OPRA tape / historical options adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from gamma_squeeze.data_sources.base import not_configured, ok
from gamma_squeeze.data_sources.registry import register


class OPRAOptionsAdapter:
    name = "options.opra"
    category = "options"

    def health(self) -> dict[str, Any]:
        root = os.getenv("OPRA_DATA_PATH", "").strip()
        return {
            "adapter": self.name,
            "configured": bool(root and Path(root).exists()) or bool(os.getenv("OPRA_API_KEY")),
            "opra_data_path": root or None,
            "hint": "Point OPRA_DATA_PATH at Databento/OPRA dumps or set OPRA_API_KEY",
        }

    def fetch_matrix(self, symbol: str, as_of: str | None = None) -> dict:
        root = os.getenv("OPRA_DATA_PATH", "").strip()
        if not root and not os.getenv("OPRA_API_KEY"):
            return not_configured(
                self.name,
                self.category,
                hint="Set OPRA_DATA_PATH (local OPRA day files) or OPRA_API_KEY",
            ).to_dict()
        # Local file probe
        if root and as_of:
            candidate = Path(root) / symbol.upper() / f"{as_of}.json"
            if candidate.is_file():
                import json

                with candidate.open() as f:
                    matrix = json.load(f)
                return ok(
                    self.name,
                    self.category,
                    symbol=symbol.upper(),
                    as_of=as_of,
                    data={"matrix": matrix},
                    meta={"origin": "opra_data_path"},
                ).to_dict()
        return ok(
            self.name,
            self.category,
            symbol=symbol.upper(),
            as_of=as_of,
            data={"matrix": None},
            meta={"status": "stub", "message": "OPRA path configured; day file not found or API not wired"},
        ).to_dict()


register("options.opra", OPRAOptionsAdapter)
