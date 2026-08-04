"""Gamma / delta exposure (GEX / DEX) from Alpaca options matrices."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GEXSnapshot:
    symbol: str
    as_of_date: str
    spot: float
    net_gex: float
    call_gex: float
    put_gex: float
    net_dex: float
    call_dex: float
    put_dex: float
    call_oi: float
    put_oi: float
    call_volume: float
    put_volume: float
    regime: str  # positive_gamma | negative_gamma | neutral_gamma
    gamma_flip: float | None
    call_wall: float | None
    put_wall: float | None


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_gamma(spot: float, strike: float, t_years: float, iv: float = 0.35, r: float = 0.04) -> float:
    if t_years <= 0 or spot <= 0 or strike <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    return _norm_pdf(d1) / (spot * iv * math.sqrt(t_years))


def _bs_delta(
    spot: float,
    strike: float,
    t_years: float,
    *,
    is_call: bool,
    iv: float = 0.35,
    r: float = 0.04,
) -> float:
    if t_years <= 0 or spot <= 0 or strike <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    if is_call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def _contracts_from_matrix(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if matrix.get("contracts"):
        return list(matrix["contracts"])
    rows: list[dict[str, Any]] = []
    for exp_block in matrix.get("by_expiration", []):
        for strike_block in exp_block.get("strikes", []):
            for side in ("call", "put"):
                contract = strike_block.get(side)
                if contract:
                    rows.append(contract)
    return rows


def compute_gex_snapshot(matrix: dict[str, Any], *, as_of_date: str = "") -> GEXSnapshot:
    """Compute net/call/put GEX and DEX from an Alpaca options matrix payload."""
    symbol = str(matrix.get("symbol", "")).upper()
    spot = float(matrix.get("underlying_price") or 0)
    contracts = _contracts_from_matrix(matrix)
    now = datetime.now(timezone.utc)

    call_gex = put_gex = 0.0
    call_dex = put_dex = 0.0
    call_oi = put_oi = 0.0
    call_vol = put_vol = 0.0
    levels: dict[float, float] = {}

    for c in contracts:
        strike = float(c.get("strike_price") or 0)
        oi = float(c.get("open_interest") or 0)
        vol = float(c.get("total_volume") or 0)
        if strike <= 0:
            continue
        cp = str(c.get("put_call", "CALL")).upper()
        is_call = cp.startswith("C")
        exp_str = c.get("expiration_date")
        t_years = 30 / 365
        if exp_str:
            try:
                exp = datetime.fromisoformat(str(exp_str).replace("Z", "+00:00"))
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                t_years = max((exp - now).total_seconds() / (365.25 * 86400), 1 / 365)
            except ValueError:
                pass
        iv = float(c.get("volatility") or 0.35)
        if iv > 5:
            iv /= 100.0
        iv = max(min(iv, 3.0), 0.05)
        gamma = float(c.get("gamma") or 0) or _bs_gamma(spot, strike, t_years, iv=iv)
        raw_delta = c.get("delta")
        try:
            delta = float(raw_delta) if raw_delta not in (None, "") else 0.0
        except (TypeError, ValueError):
            delta = 0.0
        if not delta:
            delta = _bs_delta(spot, strike, t_years, is_call=is_call, iv=iv)

        gex = gamma * oi * 100 * spot
        dex = delta * oi * 100 * spot

        if is_call:
            call_gex += gex
            call_dex += dex
            call_oi += oi
            call_vol += vol
            levels[strike] = levels.get(strike, 0.0) + gex
        else:
            put_gex += gex
            put_dex += abs(dex)
            put_oi += oi
            put_vol += vol
            levels[strike] = levels.get(strike, 0.0) - gex

    net = call_gex - put_gex
    net_dex = call_dex - put_dex
    if net > 0:
        regime = "positive_gamma"
    elif net < 0:
        regime = "negative_gamma"
    else:
        regime = "neutral_gamma"

    strikes = sorted(levels.keys())
    gamma_flip = None
    if strikes:
        for i in range(1, len(strikes)):
            if levels[strikes[i - 1]] * levels[strikes[i]] < 0:
                gamma_flip = strikes[i]
                break
    call_wall = max(levels, key=levels.get) if levels else None
    put_wall = min(levels, key=levels.get) if levels else None

    return GEXSnapshot(
        symbol=symbol,
        as_of_date=as_of_date,
        spot=spot,
        net_gex=net,
        call_gex=call_gex,
        put_gex=put_gex,
        net_dex=net_dex,
        call_dex=call_dex,
        put_dex=put_dex,
        call_oi=call_oi,
        put_oi=put_oi,
        call_volume=call_vol,
        put_volume=put_vol,
        regime=regime,
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        put_wall=put_wall,
    )


def load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_matrix_dates(root: Path, symbol: str) -> list[str]:
    sym_dir = root / symbol
    if not sym_dir.is_dir():
        return []
    return sorted(p.stem for p in sym_dir.glob("*.json") if not p.name.startswith("._"))
