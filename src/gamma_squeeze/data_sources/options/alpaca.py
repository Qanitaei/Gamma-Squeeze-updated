"""Alpaca options matrix adapter — live API first, then SSD / KV backup."""

from __future__ import annotations

import os
from typing import Any

from gamma_squeeze.config import (
    alpaca_backup_worker_url,
    alpaca_configured,
    alpaca_credentials,
    resolve_matrix_root,
)
from gamma_squeeze.data_sources.base import not_configured, ok
from gamma_squeeze.data_sources.options._common import normalize_matrix
from gamma_squeeze.data_sources.registry import register
from gamma_squeeze.ingest.alpaca_backup_client import (
    fetch_options_matrix as fetch_kv_matrix,
    health as backup_health,
    list_local_dates,
    list_matrix_dates,
    load_local_matrix,
)


class AlpacaOptionsAdapter:
    name = "options.alpaca"
    category = "options"

    def health(self) -> dict[str, Any]:
        h = backup_health()
        creds = alpaca_credentials()
        return {
            "adapter": self.name,
            "api_keys": alpaca_configured(),
            "trading_base": creds.get("trading_base"),
            "data_base": creds.get("data_base"),
            "paper": creds.get("paper"),
            "backup_worker": alpaca_backup_worker_url(),
            "backup_kv_namespace": "alpaca-options-matrix-backup",
            "backup_health": h,
            "ssd_root": str(resolve_matrix_root()),
            "primary": "alpaca_api",
            "fetch_order": ["alpaca_api", "ssd", "alpaca-options-matrix-backup"],
        }

    def fetch_matrix(self, symbol: str, as_of: str | None = None) -> dict:
        sym = symbol.upper()
        root = resolve_matrix_root()
        errors: list[str] = []

        # 1) Live / dated Alpaca API
        if alpaca_configured():
            try:
                from gamma_squeeze.ingest.alpaca_api import fetch_live_options_matrix

                live = fetch_live_options_matrix(sym, as_of=as_of)
                if live.get("contracts"):
                    matrix = normalize_matrix(live, source="alpaca", symbol=sym)
                    return ok(
                        self.name,
                        self.category,
                        symbol=sym,
                        as_of=as_of or matrix.get("as_of_date"),
                        data={"matrix": matrix},
                        meta={"origin": "alpaca_api", "n_contracts": len(live.get("contracts") or [])},
                    ).to_dict()
                errors.append("alpaca_api_empty")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"alpaca_api:{exc}")

        # 2) Local SSD matrix
        if as_of:
            local = load_local_matrix(root, sym, as_of)
            if local:
                matrix = normalize_matrix(local, source="alpaca", symbol=sym)
                return ok(
                    self.name,
                    self.category,
                    symbol=sym,
                    as_of=as_of,
                    data={"matrix": matrix},
                    meta={"origin": "ssd", "upstream_errors": errors},
                ).to_dict()
        else:
            dates = list_local_dates(root, sym) or []
            if dates:
                local = load_local_matrix(root, sym, dates[-1])
                if local:
                    matrix = normalize_matrix(local, source="alpaca", symbol=sym)
                    return ok(
                        self.name,
                        self.category,
                        symbol=sym,
                        as_of=dates[-1],
                        data={"matrix": matrix},
                        meta={"origin": "ssd", "upstream_errors": errors},
                    ).to_dict()

        # 3) KV backup worker
        try:
            matrix = fetch_kv_matrix(sym, as_of)
            matrix = normalize_matrix(matrix, source="alpaca", symbol=sym)
            return ok(
                self.name,
                self.category,
                symbol=sym,
                as_of=as_of or matrix.get("as_of_date"),
                data={"matrix": matrix},
                meta={"origin": "kv_worker", "upstream_errors": errors},
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"kv:{exc}")

        return {
            **not_configured(
                self.name,
                self.category,
                hint="Configure ALPACA_API_KEY_ID/SECRET or sync SSD/KV matrices",
            ).to_dict(),
            "success": False,
            "error": "; ".join(errors) or "no_matrix",
            "configured": alpaca_configured() or bool(os.getenv("ALPACA_BACKUP_WORKER_URL")),
        }

    def list_dates(self, symbol: str, *, prefer_local: bool = True) -> dict:
        sym = symbol.upper()
        root = resolve_matrix_root()
        local = list_local_dates(root, sym)
        remote: list[str] = []
        try:
            remote = list_matrix_dates(sym)
        except Exception as exc:  # noqa: BLE001
            return ok(
                self.name,
                self.category,
                symbol=sym,
                data={"dates": local if prefer_local else []},
                meta={"local_dates": local, "remote_error": str(exc)},
            ).to_dict()
        dates = local if (prefer_local and local) else remote
        return ok(
            self.name,
            self.category,
            symbol=sym,
            data={"dates": dates},
            meta={"local_n": len(local), "remote_n": len(remote)},
        ).to_dict()


register("options.alpaca", AlpacaOptionsAdapter)
