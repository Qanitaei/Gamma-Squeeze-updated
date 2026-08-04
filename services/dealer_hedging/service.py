from __future__ import annotations

from dataclasses import asdict
from typing import Any

from gamma_squeeze.config import resolve_matrix_root
from gamma_squeeze.dealer.hedge_demand import (
    DEALER_ESTIMATE_LABELS,
    DEALER_ESTIMATES,
    DEALER_OUTPUT_LABELS,
    DEALER_OUTPUTS,
    compute_hedge_demand_curve,
    simulate_dealer_hedging,
)
from gamma_squeeze.ingest.alpaca_backup_client import list_local_dates, load_local_matrix


def _load_options_matrix(symbol: str, as_of: str | None = None) -> tuple[dict[str, Any] | None, str | None, str]:
    """Load options matrix: local SSD → Alpaca API → alpaca-options-matrix-backup KV."""
    sym = symbol.upper()
    root = resolve_matrix_root()
    dates = list_local_dates(root, sym)
    date_used = as_of or (dates[-1] if dates else None)
    if date_used:
        matrix = load_local_matrix(root, sym, date_used)
        if matrix:
            return matrix, date_used, "ssd"

    try:
        from gamma_squeeze.data_sources.registry import get_adapter, load_all_adapters

        load_all_adapters()
        out = get_adapter("options.alpaca").fetch_matrix(sym, as_of=as_of)
        matrix = (out.get("data") or {}).get("matrix")
        if matrix:
            origin = (out.get("meta") or {}).get("origin", "alpaca")
            used = out.get("as_of") or as_of or matrix.get("as_of_date")
            return matrix, used, str(origin)
    except Exception:  # noqa: BLE001
        pass
    return None, date_used, "none"


def run_dealer_hedge(
    symbol: str,
    *,
    as_of: str | None = None,
    grid_pct: float = 0.08,
    n_points: int = 17,
    expected_move_pct: float | None = None,
) -> dict[str, Any]:
    sym = symbol.upper()
    matrix, date_used, origin = _load_options_matrix(sym, as_of=as_of)
    if not matrix:
        return {
            "symbol": sym,
            "as_of": date_used,
            "error": "no_options_matrix",
            "origin": origin,
            "curve": [],
            "estimates": list(DEALER_ESTIMATES),
            "outputs": list(DEALER_OUTPUTS),
            "hint": "Configure ALPACA_API_KEY_ID/SECRET or sync alpaca-options-matrix-backup / SSD",
        }

    sim = simulate_dealer_hedging(
        matrix,
        grid_pct=grid_pct,
        n_points=n_points,
        expected_move_pct=expected_move_pct,
        as_of=date_used,
    )
    payload = sim.to_dict()
    curve = compute_hedge_demand_curve(matrix, grid_pct=grid_pct, n_points=n_points)
    payload.update(
        {
            "symbol": sym,
            "as_of": date_used or payload.get("as_of"),
            "spot": matrix.get("underlying_price") or sim.spot,
            "curve": [asdict(p) for p in curve],
            "n_points": len(curve),
            "origin": origin,
            "success": not bool(sim.meta.get("error")),
            "estimates": list(DEALER_ESTIMATES),
            "estimate_labels": dict(DEALER_ESTIMATE_LABELS),
            "outputs": list(DEALER_OUTPUTS),
            "output_labels": dict(DEALER_OUTPUT_LABELS),
        }
    )
    if sim.meta.get("error"):
        payload["error"] = sim.meta["error"]
    return payload
