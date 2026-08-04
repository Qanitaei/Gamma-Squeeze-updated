"""Dealer positioning aggregates (PCR, walls, flip stress flags)."""

from __future__ import annotations

from typing import Any


def positioning_flags(gex_row: dict[str, Any]) -> dict[str, float]:
    """Binary / continuous stress flags used by labels and models."""
    net_gex = float(gex_row.get("net_gex") or 0.0)
    pcr = gex_row.get("pcr_oi")
    pcr_f = float(pcr) if pcr is not None else 1.0
    flip_dist = gex_row.get("flip_dist_pct")
    flip_f = float(flip_dist) if flip_dist is not None else 0.0
    call_wall_dist = gex_row.get("call_wall_dist_pct")
    cwd = float(call_wall_dist) if call_wall_dist is not None else 0.05

    fragile = 1.0 if net_gex <= 0 else 0.0
    near_flip = 1.0 if abs(flip_f) < 0.02 else 0.0
    call_crowding = 1.0 if pcr_f < 0.85 else 0.0
    wall_magnet = 1.0 if 0 < cwd < 0.04 else 0.0

    stress = 0.35 * fragile + 0.25 * near_flip + 0.25 * call_crowding + 0.15 * wall_magnet
    return {
        "flag_fragile_gamma": fragile,
        "flag_near_flip": near_flip,
        "flag_call_crowding": call_crowding,
        "flag_wall_magnet": wall_magnet,
        "positioning_stress": float(stress),
    }


def enrich_with_positioning(gex_row: dict[str, Any]) -> dict[str, Any]:
    out = dict(gex_row)
    out.update(positioning_flags(gex_row))
    return out
