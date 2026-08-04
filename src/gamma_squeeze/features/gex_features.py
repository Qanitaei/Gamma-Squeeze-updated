"""GEX/DEX feature extraction from in-platform Alpaca matrix analysis."""

from __future__ import annotations

from typing import Any

from gamma_squeeze.dealer.gex_analysis import compute_gex_snapshot


def gex_feature_row(matrix: dict[str, Any], *, as_of_date: str = "") -> dict[str, Any]:
    snap = compute_gex_snapshot(matrix, as_of_date=as_of_date)
    spot = float(snap.spot or 0.0)
    pcr_oi = (snap.put_oi / snap.call_oi) if snap.call_oi else None
    pcr_vol = (snap.put_volume / snap.call_volume) if snap.call_volume else None
    flip_dist = None
    if snap.gamma_flip is not None and spot > 0:
        flip_dist = (spot - float(snap.gamma_flip)) / spot
    call_wall_dist = None
    if snap.call_wall is not None and spot > 0:
        call_wall_dist = (float(snap.call_wall) - spot) / spot
    put_wall_dist = None
    if snap.put_wall is not None and spot > 0:
        put_wall_dist = (spot - float(snap.put_wall)) / spot

    regime_neg = 1.0 if snap.regime == "negative_gamma" else 0.0
    return {
        "symbol": snap.symbol,
        "as_of": as_of_date or snap.as_of_date,
        "spot": spot,
        "net_gex": float(snap.net_gex),
        "call_gex": float(snap.call_gex),
        "put_gex": float(snap.put_gex),
        "net_dex": float(snap.net_dex),
        "call_dex": float(snap.call_dex),
        "put_dex": float(snap.put_dex),
        "call_oi": float(snap.call_oi),
        "put_oi": float(snap.put_oi),
        "call_volume": float(snap.call_volume),
        "put_volume": float(snap.put_volume),
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
        "regime": snap.regime,
        # Canonical key used by feature store / models; keep legacy alias too.
        "regime_neg_gamma": regime_neg,
        "regime_neg_self": regime_neg,
        "gamma_flip": snap.gamma_flip,
        "call_wall": snap.call_wall,
        "put_wall": snap.put_wall,
        "flip_dist_pct": flip_dist,
        "call_wall_dist_pct": call_wall_dist,
        "put_wall_dist_pct": put_wall_dist,
        "gex_per_spot": float(snap.net_gex) / spot if spot else 0.0,
    }
